CREATE DATABASE IF NOT EXISTS medicine_db
    DEFAULT CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE medicine_db;

SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS batch_medicines;
DROP TABLE IF EXISTS patients;
DROP TABLE IF EXISTS batches;
DROP TABLE IF EXISTS drugs;
SET FOREIGN_KEY_CHECKS = 1;

CREATE TABLE IF NOT EXISTS batches (
    batch_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '批次ID',
    patient_name VARCHAR(100) NOT NULL COMMENT '病人姓名',
    patient_gender VARCHAR(20) NULL COMMENT '病人性别',
    patient_age INT NULL COMMENT '病人年龄',
    department VARCHAR(100) NULL COMMENT '科室',
    bed_no VARCHAR(50) NULL COMMENT '床号',
    medicines_json JSON NOT NULL COMMENT '本批次药品信息，例如 [{"medicine_name":"药品名","quantity":1}]',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_batches_patient_name (patient_name),
    INDEX idx_batches_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

CREATE TABLE IF NOT EXISTS drugs (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '药品ID',
    medicine_name VARCHAR(255) NOT NULL UNIQUE COMMENT '药品名称',
    sift1 MEDIUMBLOB NULL COMMENT '第1张模板图SIFT描述子',
    sift2 MEDIUMBLOB NULL COMMENT '第2张模板图SIFT描述子',
    sift3 MEDIUMBLOB NULL COMMENT '第3张模板图SIFT描述子',
    sift4 MEDIUMBLOB NULL COMMENT '第4张模板图SIFT描述子',
    sift5 MEDIUMBLOB NULL COMMENT '第5张模板图SIFT描述子',
    sift6 MEDIUMBLOB NULL COMMENT '第6张模板图SIFT描述子',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_drugs_medicine_name (medicine_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

SELECT 'two-table schema created' AS status;

-- 查看药品表结构
SHOW COLUMNS FROM drugs;

-- 查看药品表中已录入的药品及每个药品的 SIFT 模板数量
SELECT
    id,
    medicine_name,
    (sift1 IS NOT NULL) +
    (sift2 IS NOT NULL) +
    (sift3 IS NOT NULL) +
    (sift4 IS NOT NULL) +
    (sift5 IS NOT NULL) +
    (sift6 IS NOT NULL) AS sift_template_count,
    created_at,
    updated_at
FROM drugs
ORDER BY id;



SELECT
    id,
    medicine_name,
    OCTET_LENGTH(sift1) AS sift1_bytes,
    OCTET_LENGTH(sift2) AS sift2_bytes,
    OCTET_LENGTH(sift3) AS sift3_bytes,
    OCTET_LENGTH(sift4) AS sift4_bytes,
    OCTET_LENGTH(sift5) AS sift5_bytes,
    OCTET_LENGTH(sift6) AS sift6_bytes
FROM drugs
ORDER BY id;
