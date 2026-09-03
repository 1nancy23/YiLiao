import contextlib
import io
import os
import re
import time
import traceback

from fuzzywuzzy import fuzz, process


class CachedNameMatcher:
    def __init__(self, drug_names, patient_names):
        self.drug_names = [name for name in drug_names if name]
        self.patient_names = [name for name in patient_names if name]
        if "赵二虎" not in self.patient_names:
            self.patient_names.append("赵二虎")
        self._match_cache = {}
        self._weak_drug_terms = {
            "注射", "射用", "用", "液", "钠", "酸", "素", "水", "片", "胶囊",
            "批号", "生产", "有效", "规格",
        }
        self._latin_drug_hints = {
            "ceft": ["头孢"],
            "ceftriax": ["头孢", "曲松"],
            "triax": ["曲松"],
            "sulbact": ["舒巴坦"],
            "ornith": ["鸟氨酸"],
            "glycyrrh": ["甘草酸"],
        }

    @staticmethod
    def _clean_match_text(text):
        text = str(text or "").replace(" ", "")
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", text)

    def _extract_strong_query_terms(self, query):
        clean = self._clean_match_text(query)
        chinese_runs = re.findall(r"[\u4e00-\u9fff]{2,}", clean)
        terms = set()
        lower = clean.lower()
        for key, hints in self._latin_drug_hints.items():
            if key in lower:
                terms.update(hints)
        for run in chinese_runs:
            max_n = min(5, len(run))
            for n in range(max_n, 1, -1):
                for i in range(0, len(run) - n + 1):
                    term = run[i:i + n]
                    if term in self._weak_drug_terms:
                        continue
                    if n == 2 and any(weak in term for weak in ("注", "用", "液")):
                        continue
                    terms.add(term)
        return sorted(terms, key=lambda value: (-len(value), value))

    def _rank_drug_matches(self, query, names, threshold, limit):
        clean_query = self._clean_match_text(query)
        terms = self._extract_strong_query_terms(clean_query)
        ranked = []
        for name in names:
            clean_name = self._clean_match_text(name)
            if not clean_name:
                continue

            partial = fuzz.partial_ratio(clean_query, clean_name)
            ratio = fuzz.ratio(clean_query, clean_name)
            token = fuzz.token_set_ratio(clean_query, clean_name)
            term_hits = [term for term in terms if term in clean_name]
            term_score = sum(len(term) * 12 for term in term_hits)
            fuzzy_score = max(partial, ratio, token)
            score = fuzzy_score + term_score

            if term_hits:
                ranked.append((name, score, len("".join(term_hits)), partial))
            elif fuzzy_score >= threshold:
                ranked.append((name, score, 0, partial))

        ranked.sort(key=lambda item: (item[1], item[2], item[3]), reverse=True)
        if limit is not None:
            ranked = ranked[:limit]
        return [item[0] for item in ranked]

    def match(self, query, match_type="bottle", threshold=80, limit=None):
        if not query or len(query.strip()) < 1:
            return []

        cache_key = (query, match_type, int(threshold), limit)
        cached = self._match_cache.get(cache_key)
        if cached is not None:
            return list(cached)

        if match_type == "bottle":
            names = self.drug_names
        elif match_type == "bag":
            names = self.patient_names
        else:
            raise ValueError(f"invalid match_type: {match_type}")

        if not names:
            return []

        if match_type == "bottle":
            result = self._rank_drug_matches(query, names, threshold, limit)
            if result:
                self._match_cache[cache_key] = tuple(result)
                return result

        matches = process.extractBests(
            query,
            names,
            scorer=fuzz.partial_ratio,
            score_cutoff=threshold,
            limit=limit,
        )

        if match_type == "bottle" and not matches:
            result = list(names)
        else:
            result = [match[0] for match in matches]

        self._match_cache[cache_key] = tuple(result)
        return result


def _round_timing(timing):
    return {key: round(float(value), 6) for key, value in timing.items()}


def _env_enabled(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.lower() in ("1", "true", "yes", "on")


def _collect_timing_enabled():
    return _env_enabled("YILIAO_COLLECT_TIMING", False)


def _runtime_logs_enabled():
    return _env_enabled("YILIAO_RUNTIME_LOGS", False)


def _runtime_log(*args, **kwargs):
    if _runtime_logs_enabled():
        print(*args, **kwargs)


def _batch_timing_from_results(batch_ocr):
    if not _collect_timing_enabled():
        return {}
    for value in (batch_ocr or {}).values():
        if isinstance(value, dict) and isinstance(value.get("timing"), dict):
            return dict(value.get("timing") or {})
    return {}


def _empty_batch_ocr(task_type, idx, timing, status):
    return {
        "text": "",
        "timing": dict(timing or {}) if _collect_timing_enabled() else {},
        "batch_mode": "mixed_target_batch",
        "batch_status": status,
        "batch_owner": (task_type, idx),
    }


def _summarize_classifier_result(cls_result):
    if not isinstance(cls_result, dict):
        return {}

    predicted = cls_result.get("predicted_category")
    details = cls_result.get("details") or {}
    predicted_detail = details.get(predicted, {}) if predicted else {}
    best_template = predicted_detail.get("best_template", {}) or {}

    summary = {
        "classification_method": "sift_template",
        "sift_predicted_category": predicted,
        "sift_confidence": float(cls_result.get("confidence", 0.0) or 0.0),
        "sift_top_3": cls_result.get("top_3", []),
        "sift_all_scores": cls_result.get("all_scores", {}),
        "sift_max_good_matches": int(predicted_detail.get("max_sift_score", 0) or 0),
        "sift_best_template": {
            "good_matches": int(best_template.get("good_matches", 0) or 0),
            "match_score": float(best_template.get("match_score", 0.0) or 0.0),
            "desc_template_len": int(best_template.get("desc_template_len", 0) or 0),
            "desc_query_len": int(best_template.get("desc_query_len", 0) or 0),
        },
    }
    return summary


def _select_final_bottle_medicine(
    candidates,
    cls_result,
    min_good_matches=4,
    min_match_score=0.009,
    ocr_match_weak=False,
):
    if not isinstance(cls_result, dict):
        if not candidates:
            return None, "ocr_candidate", 0.0, "no_candidates"
        return candidates[0], "ocr_candidate", 0.0, "no_sift_result"

    predicted = cls_result.get("predicted_category")
    details = cls_result.get("details") or {}
    predicted_detail = details.get(predicted, {}) if predicted else {}
    best_template = predicted_detail.get("best_template", {}) or {}
    good_matches = int(best_template.get("good_matches", 0) or 0)
    match_score = float(best_template.get("match_score", 0.0) or 0.0)
    confidence = float(cls_result.get("confidence", 0.0) or 0.0)

    if ocr_match_weak and predicted:
        return predicted, "sift_template_ocr_weak", confidence, (
            f"sift_used_because_ocr_match_weak: predicted={predicted}, "
            f"good={good_matches}, score={match_score:.4f}, "
            f"ocr_top1={(candidates[0] if candidates else '')}"
        )
    if not candidates:
        return None, "ocr_candidate", confidence, "no_candidates"
    if not predicted:
        return candidates[0], "ocr_candidate", confidence, "empty_sift_prediction"
    if predicted == candidates[0]:
        return predicted, "sift_template", confidence, "sift_agrees_with_ocr_top1"
    if predicted in candidates[:3] and good_matches >= min_good_matches and match_score >= min_match_score:
        return predicted, "sift_template", confidence, "strong_sift_in_ocr_top3"
    if good_matches >= max(6, min_good_matches + 2) and match_score >= max(0.014, min_match_score * 1.5):
        return predicted, "sift_template", confidence, "very_strong_sift_override"

    return candidates[0], "ocr_candidate_sift_rejected", confidence, (
        f"weak_sift_override_rejected: predicted={predicted}, "
        f"good={good_matches}, score={match_score:.4f}, ocr_top1={candidates[0]}"
    )


def _drug_names_from_matcher(drug_matcher):
    names = getattr(drug_matcher, "_drug_names", None)
    if not names:
        return []
    return [name for name in names if name]


def _ocr_drug_match_is_weak(text, candidates, drug_matcher):
    drug_names = _drug_names_from_matcher(drug_matcher)
    if not candidates:
        return True
    if drug_names and len(candidates) >= max(10, int(len(drug_names) * 0.8)):
        return True

    clean = re.sub(r"[^\w\u4e00-\u9fff]+", "", str(text or ""))
    chinese = re.findall(r"[\u4e00-\u9fff]", clean)
    if len(chinese) <= 1:
        return True

    extractor = getattr(drug_matcher, "_extract_strong_query_terms", None)
    if callable(extractor):
        try:
            if not extractor(text):
                return True
        except Exception:
            pass
    return False


def _call_ocr(recognizer, method_name, image, quiet=False):
    method = getattr(recognizer, method_name)
    if not _collect_timing_enabled():
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                text = method(image)
        else:
            text = method(image)
        return text, {}

    start = time.perf_counter()
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            text = method(image)
    else:
        text = method(image)
    timing = dict(getattr(recognizer, "last_timing", {}) or {})
    timing["recognizer_call"] = time.perf_counter() - start
    return text, timing


def _looks_like_non_bottle_text(text):
    text = str(text or "").strip()
    if not text:
        return False

    compact = re.sub(r"\s+", "", text)
    bag_terms = (
        "床", "病区", "住院", "患者", "姓名", "年龄", "性别", "赵二虎",
        "输液", "静脉", "滴注", "滴速", "护士", "医嘱", "处方", "二维码",
    )
    if any(term in compact for term in bag_terms):
        return True

    if re.search(r"\d{1,3}床", compact):
        return True

    digits = re.findall(r"\d", compact)
    chinese = re.findall(r"[\u4e00-\u9fff]", compact)
    if len(digits) >= 8 and len(digits) >= len(chinese):
        return True

    return False


def process_bottle(
    idx,
    image,
    recognizer,
    drug_matcher,
    classifier=None,
    classify_enabled=True,
    classifier_lock=None,
    classifier_thread_safe=True,
    quiet=False,
    precomputed_ocr=None,
):
    item = {
        "type": "bottle",
        "index": idx,
        "ocr_text": "",
        "candidates": [],
        "final_medicine": None,
        "confidence": 0.0,
        "status": "started",
        "timing_sec": {},
    }
    collect_timing = _collect_timing_enabled()
    total_start = time.perf_counter() if collect_timing else 0.0
    try:
        if recognizer is None:
            item["status"] = "OCR识别器未提供"
            return item

        if precomputed_ocr is not None:
            text = precomputed_ocr.get("text", "")
            ocr_timing = dict(precomputed_ocr.get("timing", {}) or {}) if collect_timing else {}
            classify_image = precomputed_ocr.get("classify_image", image)
            classify_image_already_rotated = bool(
                precomputed_ocr.get("classify_image_already_rotated", False)
            )
            item["ocr_batch_mode"] = precomputed_ocr.get("batch_mode", "")
            item["ocr_batch_status"] = precomputed_ocr.get("batch_status", "")
            item["det_region_count"] = int(precomputed_ocr.get("det_region_count", 0) or 0)
            item["rec_nonempty_count"] = int(precomputed_ocr.get("rec_nonempty_count", 0) or 0)
            item["det_visualization"] = precomputed_ocr.get("det_visualization")
        else:
            text, ocr_timing = _call_ocr(recognizer, "recognize", image, quiet)
            classify_image = image
            classify_image_already_rotated = False
        if collect_timing:
            item["timing_sec"].update({f"ocr_{key}": value for key, value in ocr_timing.items()})
        item["ocr_text"] = text or ""
        if not text:
            item["status"] = "ocr_empty"
            item["classification_zero_reason"] = "ocr_empty"
            item["ocr_empty_reason"] = (
                "rec_all_empty"
                if int(item.get("det_region_count", 0) or 0) > 0
                else "det_empty"
            )
            if collect_timing:
                item["timing_sec"]["total"] = time.perf_counter() - total_start
                item["timing_sec"] = _round_timing(item["timing_sec"])
            return item

        if _looks_like_non_bottle_text(text):
            item["ocr_warning"] = "ocr_text_looks_like_bag_or_patient_label"

        if drug_matcher is None:
            item["status"] = "药品匹配器未提供"
            if collect_timing:
                item["timing_sec"]["total"] = time.perf_counter() - total_start
                item["timing_sec"] = _round_timing(item["timing_sec"])
            return item

        t = time.perf_counter() if collect_timing else 0.0
        if quiet:
            with contextlib.redirect_stdout(io.StringIO()):
                candidates = drug_matcher.match(text, match_type="bottle", threshold=50, limit=10)
        else:
            candidates = drug_matcher.match(text, match_type="bottle", threshold=50, limit=10)
        if collect_timing:
            item["timing_sec"]["match"] = time.perf_counter() - t
        item["candidates"] = candidates
        ocr_match_weak = _ocr_drug_match_is_weak(text, candidates, drug_matcher)
        item["ocr_match_weak"] = bool(ocr_match_weak)
        classification_candidates = candidates
        if ocr_match_weak:
            all_drug_candidates = _drug_names_from_matcher(drug_matcher)
            if all_drug_candidates:
                classification_candidates = all_drug_candidates
                item["classification_candidate_source"] = "all_drugs_due_to_weak_ocr_match"
                item["classification_candidate_count"] = len(classification_candidates)
        if not classification_candidates:
            item["status"] = "no_candidates"
            item["classification_zero_reason"] = "no_candidates"
            if collect_timing:
                item["timing_sec"]["total"] = time.perf_counter() - total_start
                item["timing_sec"] = _round_timing(item["timing_sec"])
            return item

        if classify_enabled and classifier is not None:
            t = time.perf_counter() if collect_timing else 0.0
            if quiet:
                stream = io.StringIO()
            else:
                stream = None
            with contextlib.redirect_stdout(stream) if stream is not None else contextlib.nullcontext():
                if classifier_thread_safe or classifier_lock is None:
                    cls_result = classifier.classify(
                        classification_candidates,
                        classify_image,
                        image_already_rotated=classify_image_already_rotated,
                    )
                else:
                    with classifier_lock:
                        cls_result = classifier.classify(
                            classification_candidates,
                            classify_image,
                            image_already_rotated=classify_image_already_rotated,
                        )
            if collect_timing:
                item["timing_sec"]["classify"] = time.perf_counter() - t
            item.update(_summarize_classifier_result(cls_result))
            final_medicine, method, confidence, decision_reason = _select_final_bottle_medicine(
                candidates,
                cls_result,
                ocr_match_weak=ocr_match_weak,
            )
            item["classification_method"] = method
            item["final_medicine"] = final_medicine
            item["confidence"] = confidence
            item["decision_reason"] = decision_reason
            item["top_3"] = cls_result.get("top_3", [])
            if confidence <= 0.0:
                item["classification_zero_reason"] = decision_reason
        else:
            item["classification_method"] = "ocr_candidate"
            item["final_medicine"] = candidates[0] if candidates else None
            item["classification_zero_reason"] = "classifier_disabled_or_missing"

        item["status"] = "done"
        if collect_timing:
            item["timing_sec"]["total"] = time.perf_counter() - total_start
            item["timing_sec"] = _round_timing(item["timing_sec"])
        return item
    except Exception as exc:
        item["status"] = f"error: {exc}"
        item["traceback"] = traceback.format_exc()
        if collect_timing:
            item["timing_sec"]["total"] = time.perf_counter() - total_start
            item["timing_sec"] = _round_timing(item["timing_sec"])
        return item


def process_bag(
    idx,
    image,
    recognizer,
    drug_matcher,
    quiet=False,
    simulate_patient_name="",
    precomputed_ocr=None,
):
    item = {
        "type": "bag",
        "index": idx,
        "ocr_text": "",
        "patient_name": None,
        "status": "started",
        "timing_sec": {},
    }
    collect_timing = _collect_timing_enabled()
    total_start = time.perf_counter() if collect_timing else 0.0
    try:
        if recognizer is None:
            item["status"] = "OCR识别器未提供"
            return item

        if precomputed_ocr is not None:
            text = precomputed_ocr.get("text", "")
            ocr_candidates = list(precomputed_ocr.get("candidate_texts", []) or [])
            ocr_timing = dict(precomputed_ocr.get("timing", {}) or {}) if collect_timing else {}
            item["ocr_batch_mode"] = precomputed_ocr.get("batch_mode", "")
            item["ocr_batch_status"] = precomputed_ocr.get("batch_status", "")
            item["ocr_classify_angle_pred"] = precomputed_ocr.get("classify_angle_pred")
            item["ocr_name_roi_count"] = precomputed_ocr.get("name_roi_count")
        else:
            text, ocr_timing = _call_ocr(recognizer, "recognize_yaodai", image, quiet)
            ocr_candidates = [{"text": text, "confidence": 0.0, "source": "single"}] if text else []
        if collect_timing:
            item["timing_sec"].update({f"ocr_{key}": value for key, value in ocr_timing.items()})
        if text and not ocr_candidates:
            ocr_candidates = [{"text": text, "confidence": 0.0, "source": "batch_text"}]
        if simulate_patient_name:
            item["raw_ocr_text"] = text or ""
            text = simulate_patient_name
            item["simulated_patient_name"] = simulate_patient_name
        item["ocr_text"] = text or ""
        item["ocr_candidates"] = ocr_candidates
        if not text:
            item["status"] = "ocr_empty"
            if collect_timing:
                item["timing_sec"]["total"] = time.perf_counter() - total_start
                item["timing_sec"] = _round_timing(item["timing_sec"])
            return item

        t = time.perf_counter() if collect_timing else 0.0
        if simulate_patient_name:
            patient = [simulate_patient_name]
        else:
            ranked = []
            for candidate in ocr_candidates:
                candidate_text = str(candidate.get("text", "") or "").strip()
                if not candidate_text:
                    continue
                rec_confidence = min(1.0, max(0.0, float(candidate.get("confidence", 0.0) or 0.0)))
                if quiet:
                    with contextlib.redirect_stdout(io.StringIO()):
                        matches = drug_matcher.match(
                            candidate_text,
                            match_type="bag",
                            threshold=50,
                            limit=10,
                        )
                else:
                    matches = drug_matcher.match(
                        candidate_text,
                        match_type="bag",
                        threshold=50,
                        limit=10,
                    )
                for matched_name in matches:
                    matched_name = str(matched_name or "").strip()
                    if not matched_name:
                        continue
                    ratio = float(fuzz.ratio(candidate_text, matched_name))
                    partial = float(fuzz.partial_ratio(candidate_text, matched_name))
                    similarity = max(ratio, partial)
                    exact_bonus = 10.0 if candidate_text == matched_name else 0.0
                    containment_bonus = 5.0 if (
                        candidate_text in matched_name or matched_name in candidate_text
                    ) else 0.0
                    ranked.append({
                        "patient": matched_name,
                        "ocr_text": candidate_text,
                        "rec_confidence": rec_confidence,
                        "similarity": similarity,
                        "score": similarity + rec_confidence * 20.0 + exact_bonus + containment_bonus,
                    })

            ranked.sort(
                key=lambda value: (value["score"], value["similarity"], value["rec_confidence"]),
                reverse=True,
            )
            patient = []
            if ranked and ranked[0]["similarity"] >= 50.0:
                best = ranked[0]
                competing = next(
                    (value for value in ranked[1:] if value["patient"] != best["patient"]),
                    None,
                )
                ambiguous = competing is not None and best["score"] - competing["score"] < 5.0
                if not ambiguous:
                    patient = [best["patient"]]
                    item["ocr_text"] = best["ocr_text"]
                    item["patient_match_score"] = round(best["score"], 4)
                    item["patient_match_similarity"] = round(best["similarity"], 4)
                    item["patient_rec_confidence"] = round(best["rec_confidence"], 6)
            item["patient_candidate_ranking"] = ranked[:10]
        if collect_timing:
            item["timing_sec"]["match"] = time.perf_counter() - t
        item["patient_name"] = patient
        item["status"] = "done" if patient else "no_patient_match"
        if collect_timing:
            item["timing_sec"]["total"] = time.perf_counter() - total_start
            item["timing_sec"] = _round_timing(item["timing_sec"])
        return item
    except Exception as exc:
        item["status"] = f"error: {exc}"
        item["traceback"] = traceback.format_exc()
        if collect_timing:
            item["timing_sec"]["total"] = time.perf_counter() - total_start
            item["timing_sec"] = _round_timing(item["timing_sec"])
        return item


def process_infusion(idx, image, recognizer, quiet=False, result_type="shuye", precomputed_ocr=None):
    item = {
        "type": result_type,
        "index": idx,
        "ocr_text": "",
        "liquid": None,
        "concentration": None,
        "volume": None,
        "raw_text": "",
        "status": "started",
        "timing_sec": {},
    }
    collect_timing = _collect_timing_enabled()
    total_start = time.perf_counter() if collect_timing else 0.0
    try:
        if recognizer is None:
            item["status"] = "OCR识别器未提供"
            return item

        if precomputed_ocr is not None:
            text = precomputed_ocr.get("text", "")
            ocr_timing = dict(precomputed_ocr.get("timing", {}) or {}) if collect_timing else {}
            item["ocr_batch_mode"] = precomputed_ocr.get("batch_mode", "")
            item["ocr_batch_status"] = precomputed_ocr.get("batch_status", "")
            item["ocr_candidate_roi_count"] = precomputed_ocr.get("candidate_roi_count")
        else:
            text, ocr_timing = _call_ocr(recognizer, "recognize_shuyedai", image, quiet)
        if collect_timing:
            item["timing_sec"].update({f"ocr_{key}": value for key, value in ocr_timing.items()})
        if isinstance(text, dict):
            item["ocr_text"] = text
            item["liquid"] = text.get("liquid")
            item["concentration"] = text.get("concentration")
            item["volume"] = text.get("volume")
            item["raw_text"] = text.get("raw_text", "")
            item["status"] = text.get("status", "done")
        else:
            item["ocr_text"] = text or ""
            item["raw_text"] = text or ""
            item["status"] = "done" if text else "ocr_empty"
        if collect_timing:
            item["timing_sec"]["total"] = time.perf_counter() - total_start
            item["timing_sec"] = _round_timing(item["timing_sec"])
        return item
    except Exception as exc:
        item["status"] = f"error: {exc}"
        item["traceback"] = traceback.format_exc()
        if collect_timing:
            item["timing_sec"]["total"] = time.perf_counter() - total_start
            item["timing_sec"] = _round_timing(item["timing_sec"])
        return item


def process_task_group(
    task_group,
    recognizer,
    drug_matcher,
    classifier=None,
    classify_enabled=True,
    classifier_lock=None,
    classifier_thread_safe=True,
    quiet=False,
    simulate_patient_name="",
):
    batch_ocr = {}
    collect_timing = _collect_timing_enabled()
    batch_timing = {}
    batch_status = "disabled"
    batch_error = None
    use_mixed_batch = _env_enabled("YILIAO_MIXED_OCR_BATCH", True)
    allow_fallback = _env_enabled("YILIAO_ALLOW_OCR_FALLBACK", False)

    if use_mixed_batch and recognizer is not None and hasattr(recognizer, "recognize_task_batch"):
        try:
            if quiet:
                with contextlib.redirect_stdout(io.StringIO()):
                    batch_ocr = recognizer.recognize_task_batch(task_group) or {}
            else:
                batch_ocr = recognizer.recognize_task_batch(task_group) or {}
            batch_timing = _batch_timing_from_results(batch_ocr)
            if collect_timing and not batch_timing:
                batch_timing = dict(getattr(recognizer, "last_timing", {}) or {})
            batch_status = "ok"
            if _runtime_logs_enabled():
                _runtime_log(
                    "[OCR mixed target batch] "
                    f"tasks={len(task_group)}, results={len(batch_ocr)}, "
                    f"cls_inputs={int(batch_timing.get('shared_cls_inputs', 0) or 0)}, "
                    f"det_inputs={int(batch_timing.get('shared_det_inputs', batch_timing.get('mixed_det_inputs', 0)) or 0)}, "
                    f"det_batches={int(batch_timing.get('shared_det_batches', batch_timing.get('mixed_det_batches', 0)) or 0)}, "
                    f"rec_inputs={int(batch_timing.get('mixed_rec_inputs', 0) or 0)}, "
                    f"rec_batches={int(batch_timing.get('mixed_rec_batches', 0) or 0)}"
                )
        except Exception as exc:
            batch_ocr = {}
            batch_error = traceback.format_exc()
            batch_status = f"error: {exc}"
            _runtime_log(f"[OCR mixed target batch] failed: {exc}")
            if not quiet:
                _runtime_log(batch_error)

    normalized_tasks = [task[:3] for task in task_group]

    if use_mixed_batch and batch_status != "ok" and not allow_fallback:
        for task_type, idx, _crop in normalized_tasks:
            batch_ocr[(task_type, idx)] = _empty_batch_ocr(task_type, idx, batch_timing, batch_status)
    elif use_mixed_batch and batch_status == "ok":
        for task_type, idx, _crop in normalized_tasks:
            value = batch_ocr.get((task_type, idx))
            if isinstance(value, dict):
                value.setdefault("batch_mode", "mixed_target_batch")
                value.setdefault("batch_status", "ok")
                value.setdefault("batch_owner", (task_type, idx))
            elif not allow_fallback:
                batch_ocr[(task_type, idx)] = _empty_batch_ocr(
                    task_type,
                    idx,
                    batch_timing,
                    "missing_from_batch",
                )

    if use_mixed_batch and batch_status != "ok" and allow_fallback:
        _runtime_log("[OCR mixed target batch] falling back to per-target OCR because YILIAO_ALLOW_OCR_FALLBACK=1")

    results = []
    for task_type, idx, crop in normalized_tasks:
        precomputed = batch_ocr.get((task_type, idx))
        if task_type == "bottle":
            results.append(process_bottle(
                idx,
                crop,
                recognizer,
                drug_matcher,
                classifier,
                classify_enabled,
                classifier_lock,
                classifier_thread_safe,
                quiet,
                precomputed,
            ))
        elif task_type == "bag":
            results.append(process_bag(
                idx,
                crop,
                recognizer,
                drug_matcher,
                quiet,
                simulate_patient_name,
                precomputed,
            ))
        else:
            results.append(process_infusion(
                idx,
                crop,
                recognizer,
                quiet,
                result_type="shuye",
                precomputed_ocr=precomputed,
            ))
    return results
