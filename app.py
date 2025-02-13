import os
import json
import time
import uuid
from uuid import UUID
from loguru import logger
from datetime import datetime
from dotenv import load_dotenv
from fastapi import BackgroundTasks
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, func
from sqlalchemy.orm import sessionmaker
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from typing import List, Optional, Dict, Any
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base

from main import run_consumers

# CLI command: uvicorn app:app --reload --log-level debug
load_dotenv(".env")
app = FastAPI(title="Task Execution API",
              description="API to execute and retrieve results from the FATEC tool.",
              version="1.0.0",
              contact={
                  "name": "Carlos Silva",
                  "email": "carlos.silva@inesctec.pt",
              })

# PostgreSQL connection
DATABASE_URL = f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"\
               f"@{os.environ['POSTGRES_HOST']}:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_NAME']}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Task(Base):
    __tablename__ = "tasks"
    id = Column(String, primary_key=True)
    status = Column(String, default="pending")
    result = Column(JSONB, nullable=True)
    parameters = Column(JSONB)
    launch_time = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# Input schema
class TaskRequest(BaseModel):
    partner: str = Field(..., min_length=1, example="coopernico", description="The name of the partner.")
    consumers: List[str] = Field(..., min_items=1, example=["test", "test1", "test2"], description="List of consumers.")
    city: str = Field(..., min_length=1, example="Porto", description="The city where the consumers are located.")
    aggregated: int = Field(..., ge=0, le=1, example=1, description="Whether to aggregate consumers (0 = No, 1 = Yes).")


class TaskResponse(BaseModel):
    task_id: str
    launch_time: str
    status: str
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]]
    last_updated: str


# Task to execute asynchronously
def execute_task(task_id: str, task_request: TaskRequest):
    db = SessionLocal()
    try:
        # Prepare folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        directory_name = f"results/exec_{timestamp}_{task_id}"
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        logger.add(f"{directory_name}/out.log", level="DEBUG", colorize=False, backtrace=True, diagnose=True)
        inputs = {'partner': task_request.partner, 'consumers': task_request.consumers,
                  'city': task_request.city, 'aggregated': task_request.aggregated}
        json.dump(inputs, open(f"{directory_name}/inputs.json", 'w'), indent=4)

        # Run task
        result_data = run_consumers(task_request.partner, task_request.city, task_request.consumers,
                                    task_request.aggregated, plot=1, dir_name=directory_name)
        logger.debug(f"Task execution completed.")

        # If successful, update task to "completed"
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "completed"
            task.result = result_data
            task.last_updated = datetime.utcnow()
            db.commit()

    except Exception as e:
        # If an error occurs, update task to "failed"
        logger.exception(f"Error in task execution")
        task = db.query(Task).filter(Task.id == task_id).first()
        if task:
            task.status = "failed"
            task.result = {"error": str(e)}
            task.last_updated = datetime.utcnow()
            db.commit()

    finally:
        db.close()


# API Endpoint to execute a task
@app.post("/execute")
def execute(task_request: TaskRequest, background_tasks: BackgroundTasks):
    try:
        task_id = str(uuid.uuid4())

        db = SessionLocal()

        # Ensure valid JSON storage
        try:
            parameters_json = task_request.dict()
        except Exception as e:
            logger.error(f"Invalid input format: {e}")
            raise HTTPException(status_code=400, detail="Invalid input format")

        # Insert pending task
        new_task = Task(
            id=task_id,
            status="pending",
            result=None,
            parameters=parameters_json,
            launch_time=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        db.add(new_task)
        db.commit()

        # Run FATEC asynchronously
        background_tasks.add_task(execute_task, task_id, task_request)

        return {"task_id": task_id, "status": "pending"}

    except HTTPException as e:
        raise e  # Forward known errors

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

    finally:
        db.close()


# API Endpoint to check task result
@app.get("/result/{task_id}", response_model=TaskResponse)
def get_result(task_id: str):
    db = SessionLocal()

    # Validate task_id format before querying the DB
    try:
        task_uuid = UUID(task_id, version=4)
    except ValueError:
        db.close()
        raise HTTPException(status_code=400, detail="Invalid task_id format")

    # Query DB
    task = db.query(Task).filter(Task.id == str(task_uuid)).first()
    if not task:
        db.close()
        raise HTTPException(status_code=404, detail="Task ID not found")

    try:
        # Ensure parameters and results are valid JSON
        inputs = task.parameters if isinstance(task.parameters, dict) else json.loads(task.parameters)
        outputs = task.result if isinstance(task.result, dict) else json.loads(task.result) if task.result else None
    except json.JSONDecodeError as e:
        db.close()
        logger.error(f"Error parsing JSON from DB for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Error processing task data")

    response_data = {
        "task_id": task.id,
        "launch_time": task.launch_time.isoformat(),
        "status": task.status,
        "inputs": inputs,
        "outputs": outputs,
        "last_updated": task.last_updated.isoformat(),
    }

    db.close()
    logger.debug(f"Returning response: {response_data}")
    return response_data
