-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Jun 13, 2024 at 04:40 AM
-- Server version: 10.4.27-MariaDB
-- PHP Version: 8.2.0

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `attendance_system`
--

-- --------------------------------------------------------

--
-- Table structure for table `home_emp_data`
--

CREATE TABLE `home_emp_data` (
  `id` bigint(20) NOT NULL,
  `name` varchar(100) NOT NULL,
  `img` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `home_emp_data`
--

INSERT INTO `home_emp_data` (`id`, `name`, `img`) VALUES
(1, 'emp_1', 'home/emp_imgs/emp_1.jpg'),
(2, 'emp_2', 'home/emp_imgs/emp_2.jpg'),
(3, 'emp_3', 'home/emp_imgs/emp_3.jpg'),
(4, 'emp_4', 'home/emp_imgs/emp_4.jpg'),
(6, 'Alex\r\n', 'home/emp_imgs/Alex.png'),
(7, 'Alvin\r\n', 'home/emp_imgs/Alvin.png'),
(8, 'Gavic\r\n', 'home/emp_imgs/Gavic.png'),
(9, 'Matthew\r\n', 'home/emp_imgs/Matthew.png'),
(10, 'Paul\r\n', 'home/emp_imgs/Paul.png'),
(11, 'Ziv\r\n', 'home/emp_imgs/Ziv.png'),
(12, 'Irene', 'home/emp_imgs/Irene.png');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `home_emp_data`
--
ALTER TABLE `home_emp_data`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `home_emp_data`
--
ALTER TABLE `home_emp_data`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=13;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
