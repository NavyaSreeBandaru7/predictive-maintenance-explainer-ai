# predictive-maintenance-explainer-ai

This project aims to develop a Predictive Maintenance Explainer AI system that combines traditional machine learning with cutting-edge language models to provide maintenance technicians with insightful explanations for equipment failure predictions. The system allows technicians to ask natural language questions about the predictive model's outputs and receives human-understandable explanations.

## Project Overview

The Predictive Maintenance Explainer AI project consists of the following main components:

1. Predictive Model: A machine learning model trained on equipment sensor data and maintenance history to predict potential failures.
2. Explainable AI (XAI) Model: An XAI model that generates explanations for the predictive model's outputs, providing insights into why a particular prediction was made.
3. LLM-Powered Explainer Interface: A conversational interface powered by a fine-tuned language model that allows maintenance technicians to ask questions about the predictive model's outputs and receive generated explanations.
4. User Interface: A web-based user interface for maintenance technicians to interact with the Explainer AI system.

## Dataset

The project utilizes the "Turbofan Engine Degradation Simulation Data Set" from NASA, which contains simulated sensor readings for aircraft engines under various operating conditions and failure modes. The dataset can be downloaded using the provided `download_dataset.py` script.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository:
