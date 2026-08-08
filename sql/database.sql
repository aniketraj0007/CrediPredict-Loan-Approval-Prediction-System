-- ========================================================
-- CrediPredict - Database Creation Script
-- Description: Creates the database for CrediPredict system
-- Author: B.Tech Data Science Student
-- ========================================================

-- Drop database if it already exists to allow clean re-runs
DROP DATABASE IF EXISTS credipredict_db;

-- Create database
CREATE DATABASE credipredict_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

-- Switch to the newly created database
USE credipredict_db;
