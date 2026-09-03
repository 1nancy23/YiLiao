SHOW DATABASES ;

USE medicine_db;

SHOW tables ;

DROP table drugs;

TRUNCATE TABLE drugs;

CREATE TABLE IF NOT EXISTS drugs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    medicine_name VARCHAR(255) NOT NULL UNIQUE,
    sift1 MEDIUMBLOB,
    sift2 MEDIUMBLOB,
    sift3 MEDIUMBLOB,
    sift4 MEDIUMBLOB,
    sift5 MEDIUMBLOB,
    sift6 MEDIUMBLOB,
    deep_avg MEDIUMBLOB
);

-- 查询所有药瓶
SELECT * FROM drugs;

-- 查看数据库占用大小
SELECT
    table_name AS '表名',
    engine AS '引擎',
    table_rows AS '行数',
    (data_length + index_length) / 1024 / 1024 AS '大小 (MB)'
FROM information_schema.tables
WHERE table_schema = 'medicine_db'
ORDER BY (data_length + index_length) DESC;

-- 创建病人数据表
CREATE TABLE IF NOT EXISTS patients (
    patient_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '病人ID',
    name VARCHAR(100) NOT NULL COMMENT '病人姓名',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS batches (
    batch_id INT AUTO_INCREMENT PRIMARY KEY COMMENT '批次ID',
    patient_id INT NOT NULL COMMENT '所属病人',
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS batch_medicines (
    id INT AUTO_INCREMENT PRIMARY KEY,
    batch_id INT NOT NULL COMMENT '所属批次',
    medicine_id INT NOT NULL COMMENT '药品ID（关联drugs.id）',
    quantity INT COMMENT '数量',
    FOREIGN KEY (batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE,
    FOREIGN KEY (medicine_id) REFERENCES drugs(id) ON DELETE CASCADE,
    UNIQUE KEY unique_batch_medicine (batch_id, medicine_id)
);

INSERT into patients(name, created_at) VALUES ('魏理想',null);
INSERT into batches(patient_id) VALUES (1);
INSERT into batch_medicines(batch_id, medicine_id, quantity) VALUES (1,11,1);

SELECT * from patients;

SELECT * from batches;

SELECT * from batch_medicines;

SELECT * from drugs;