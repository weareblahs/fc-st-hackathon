# What's this?

This is a project for a hackathon - which utilizes data science, frontend dev, backend dev, API dev and deployment skills, featuring prediction, data analysis and others.

## Project Context

Fleet operators generate large volumes of raw GPS and IoT telemetry data each day. While this data is rich, it is not directly actionable without transformation. As a result, companies struggle to make real-time decisions, identify inefficiencies, and optimize route performance.

This project addresses that challenge by converting raw telemetry into operational intelligence for delivery and transport fleets. The system provides:

- **Route Efficiency Scoring** based on speed patterns, idle durations, distance, and fuel-related indicators  
- **Idle Time Detection** to highlight unproductive periods and reduce wasted operational hours  
- **Fuel Consumption Insights** to support cost-saving and sustainability strategies  
- **Driver and Vehicle Performance Metrics** revealing safety and efficiency trends  
- **Predictive Analytics** forecasting route efficiency using machine-learning models  
- **Automated Visual Reports** that summarize key metrics for quick managerial decisions  

The platform integrates a **React dashboard**, **Flask backend**, and **FastAPI prediction engine**, forming a full end-to-end analytics system that transforms raw telemetry into actionable insights for smarter, greener fleet operations.

# How to run this project

## STEP 0: Before you continue with everything...

Generating data might take a long time. For this, a special endpoint on the Flask server has been set up for this purpose - but we'll go through these steps later on. For this project, you'll need:

- Node.js installation
- Python
- Git

## STEP 1: Clone the repository

Assuming that you have `git` installed, run the below command:

```bash
git clone https://github.com/weareblahs/fc-st-hackathon
cd fc-st-hackathon
```

### Notes

There are 3 servers this project needs:

- Frontend (Vite + JavaScript React, port 5173),
- Backend (Flask, port 5000),
- Backend (FastAPI prediction, port 8000)

...but all must be installed and run before you can begin using this web app.

> [!TIP]
> It is recommended that you run this project from 3 separate Terminal / Command Prompt sessions - which allows you to run commands simultaneously. This is what the project needs - one session for each server.

## STEP 2: Installing required packages for all the servers

### Frontend

Install all the packages with the NPM package manager.

```bash
cd frontend
npm i
```

### Flask backend

Install all the packages with the `pip` package manager.

```bash
cd flask_api
pip install -r requirements.txt
```

### FastAPI Backend

Install all the packages with the `pip` package manager.

```bash
cd prediction_api
pip install -r requirements.txt
```

## STEP 2.5: Configuring data

Assuming that you have the datasets (for this hackathon), extract the files and copy all the CSV files to `/flask_api/data/extracted_data`. After copying, start the Flask server by running the following:

```bash
py app.py
```

Then go to this URL in your browser (or tools such as cURL and Postman):

```
http://localhost:5000/generate_eda
```

> [!TIP]
> After you access this URL, the browser should load for a long time. This depends on the specifications of the computer - it should take around 3-5 minutes for recent computers. Go back to the terminal window where Flask is running to see readable processing logs.

Do not close this server yet! It is still being used for the frontend later on.

## STEP 3: Start the frontend and the FastAPI server

As you have configured all the server's packages, it is time to run the servers. First off, run the frontend server:

```bash
cd frontend # if you're in this directory, you can skip this step
npm run dev
```

Then run the prediction server:

```bash
cd prediction_api # if you're in this directory, you can skip this step
py app.py
```

## STEP 4: See the dashboard

Open the following URL in your browser:

```
http://localhost:5173
```

## Credits:
@[weareblahs](https://github.com/weareblahs) - Lead frontend, backend, API developer, Devops engineer

[@ThaddeusTeh2](https://github.com/ThaddeusTeh2) - team lead, project manager, data analyst, ML engineer, Presentation Lead

[@yee-ling](https://github.com/yee-ling) - Business Analyst, Technical Writer, Strategic Analyst

[@ainulmazwan](https://github.com/ainulmazwan) - Business Analyst, Technical Writer, Strategic Analyst
