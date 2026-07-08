CREATE DATABASE IF NOT EXISTS medicine_db
    DEFAULT CHARACTER SET utf8
    COLLATE utf8_general_ci;

USE medicine_db;

SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS batch_medicines;
DROP TABLE IF EXISTS patients;
DROP TABLE IF EXISTS batches;
DROP TABLE IF EXISTS drugs;
SET FOREIGN_KEY_CHECKS = 1;

CREATE TABLE IF NOT EXISTS batches (
    patient_name VARCHAR(100) NOT NULL COMMENT '病人姓名',
    medicines_json LONGTEXT NOT NULL COMMENT '药瓶信息，保存JSON字符串，例如 [{"medicine_name":"药品名","quantity":1}]',
    INDEX idx_batches_patient_name (patient_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;

CREATE TABLE IF NOT EXISTS drugs (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '药品ID',
    medicine_name VARCHAR(255) NOT NULL UNIQUE COMMENT '药品名称',
    sift1 MEDIUMBLOB NULL COMMENT '第1张模板图SIFT描述子',
    sift2 MEDIUMBLOB NULL COMMENT '第2张模板图SIFT描述子',
    sift3 MEDIUMBLOB NULL COMMENT '第3张模板图SIFT描述子',
    sift4 MEDIUMBLOB NULL COMMENT '第4张模板图SIFT描述子',
    sift5 MEDIUMBLOB NULL COMMENT '第5张模板图SIFT描述子',
    sift6 MEDIUMBLOB NULL COMMENT '第6张模板图SIFT描述子',
    INDEX idx_drugs_medicine_name (medicine_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_general_ci;

-- 手动录入测试批次数据：病人名称 + 药瓶信息
-- 说明：当前两表结构中不再单独创建 patients / batch_medicines 表，
--      病人名称和药瓶清单直接保存在 batches 表中。
INSERT INTO batches
    (patient_name, medicines_json)
VALUES
    (
        '赵二虎',
        '[{"medicine_name":"氯化钠注射液(生理盐水)","specification":"250ml","dose":"250ml","frequency":"qd"},{"medicine_name":"银杏叶提取物注射液","specification":"20ml","dose":"20ml","frequency":"qd"}]'
    ),
    (
        '魏理想',
        '[{"medicine_name":"0.9%氯化钠注射液(RD)","specification":"100ml","dose":"100ml","frequency":"QD","batch_no":"01批"},{"medicine_name":"奥美拉唑钠","specification":"40mg","dose":"40mg","frequency":"QD","batch_no":"01批"}]'
    );

SELECT 'two-table schema created' AS status;

-- 查询病人信息表结构
SHOW COLUMNS FROM batches;

-- 查询病人信息表数据
SELECT
    patient_name,
    medicines_json
FROM batches
ORDER BY patient_name;

-- 查询药瓶特征表结构
SHOW COLUMNS FROM drugs;

-- 查询药瓶特征表数据概览：已录入药品及每个药品的 SIFT 模板数量
SELECT
    id,
    medicine_name,
    (sift1 IS NOT NULL) +
    (sift2 IS NOT NULL) +
    (sift3 IS NOT NULL) +
    (sift4 IS NOT NULL) +
    (sift5 IS NOT NULL) +
    (sift6 IS NOT NULL) AS sift_template_count
FROM drugs
ORDER BY id;
