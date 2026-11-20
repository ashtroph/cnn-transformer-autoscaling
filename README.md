# Predictive and Uncertainty-Aware Autoscaling Using CNN–Transformer Workload Forecasting

## Overview

This project explores the use of deep learning for predictive resource autoscaling in cloud environments.

Traditional reactive autoscaling systems respond to workload changes after they occur. This can lead to resource shortages during sudden traffic spikes and unnecessary resource allocation during periods of low demand.

To address this problem, this project implements a **CNN–Transformer workload forecasting model** that predicts incoming workload and uses those predictions to make proactive replica-scaling decisions.

The project also introduces an uncertainty-aware autoscaling mechanism that adjusts predicted demand using a safety factor to reduce the risk of under-provisioning.

The system is evaluated through workload-driven autoscaling simulations using HTTP request traces from the 1998 FIFA World Cup.

## Research Objectives

- Forecast future workload using a CNN–Transformer architecture.
- Compare predictive autoscaling against a reactive autoscaling baseline.
- Develop an uncertainty-aware scaling mechanism to account for forecasting errors.
- Evaluate resource utilization, overload, and scaling behaviour.
- Investigate the trade-off between service capacity and resource consumption.

## System Architecture

The proposed system consists of three main components:

1. **Workload Preprocessing:** Raw HTTP request traces are converted into one-minute workload measurements
