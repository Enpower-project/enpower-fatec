# FATEC - Flexibility Assessment tool for Energy Communities

FATEC is a tool designed to assess the flexibility of participants of energy communities. It processes data and acquires weather information from external providers, and uses a causality model to compute the price elasticity of demand of consumers. The tool includes a FastAPI application to execute tasks asynchronously and store results in a PostgreSQL database.

## Features

- Data preprocessing
- Weather data acquisition using the Open-Meteo API
- Asynchronous task execution with FastAPI
- PostgreSQL database integration

## Requirements

- Python 3.8+
- PostgreSQL
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
    ```sh
    git clone https://gitlab.inesctec.pt/carlos.silva/enpower-fatec.git
    cd enpower-fatec
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up the PostgreSQL database and update the `.env` file with your database credentials.

## Usage

1. Start the FastAPI application:
    ```sh
    uvicorn app:app --reload --log-level debug
    ```

2. Use the following endpoints:

    - **Execute a task**: `POST /execute`
        - Request body:
            ```json
            {
                "partner": "partner_name",
                "consumers": ["consumer1", "consumer2"],
                "city": "city_name",
                "aggregated": 1
            }
            ```

    - **Get task result**: `GET /result/{task_id}`
      - Request parameters:
            - `task_id`: Task ID returned by the `/execute` endpoint

## Project Structure

- `app.py`: Main FastAPI application file.
- `main.py`: Contains the `run_consumers` function to execute tasks.
- `causality/`: Contains the causality analysis logic.
- `processing/`: Contains the data processing logic.
- `data/`: Contains the data of the pilot projects.
- `results/`: Contains the results of the analysis.

## License

This project is licensed under the MIT License.
