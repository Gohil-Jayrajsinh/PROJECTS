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
-- Table structure for table `home_data`
--

CREATE TABLE `home_data` (
  `id` bigint(20) NOT NULL,
  `date` varchar(50) NOT NULL,
  `name` varchar(100) NOT NULL,
  `enter_time` varchar(50) NOT NULL,
  `exit_time` varchar(50) NOT NULL,
  `exit_date` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `home_data`
--

INSERT INTO `home_data` (`id`, `date`, `name`, `enter_time`, `exit_time`, `exit_date`) VALUES
(127, '07/06/2024', 'Alex\r\n', '19:02:01', '19:47:30', '07/06/2024'),
(128, '07/06/2024', 'Alvin\r\n', '19:02:02', '19:47:55', '07/06/2024'),
(129, '07/06/2024', 'Ziv\r\n', '19:02:03', '19:47:32', '07/06/2024'),
(130, '07/06/2024', 'Matthew\r\n', '19:03:30', '19:47:54', '07/06/2024'),
(131, '07/06/2024', 'Gavic\r\n', '19:03:35', '19:47:58', '07/06/2024'),
(132, '07/06/2024', 'Paul\r\n', '19:03:52', '19:48:12', '07/06/2024'),
(133, '07/06/2024', 'Irene', '19:04:15', '19:48:36', '07/06/2024'),
(134, '13/06/2024', 'Alex\r\n', '06:55:28', '07:31:52', '13/06/2024'),
(135, '13/06/2024', 'Ziv\r\n', '06:55:33', '07:31:57', '13/06/2024'),
(136, '13/06/2024', 'Gavic\r\n', '06:55:43', '07:31:49', '13/06/2024'),
(137, '13/06/2024', 'Matthew\r\n', '06:55:57', '07:32:20', '13/06/2024'),
(138, '13/06/2024', 'Alvin\r\n', '06:55:57', '07:31:56', '13/06/2024'),
(139, '13/06/2024', 'Paul\r\n', '06:56:17', '07:32:41', '13/06/2024'),
(140, '13/06/2024', 'Irene', '06:56:40', '07:33:03', '13/06/2024');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `home_data`
--
ALTER TABLE `home_data`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `home_data`
--
ALTER TABLE `home_data`
  MODIFY `id` bigint(20) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=141;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
