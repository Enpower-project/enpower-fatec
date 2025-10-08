import os
import json
import time
import uuid
from uuid import UUID
from loguru import logger
from datetime import datetime
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

import settings
from main import run_consumers
from clusters import run_clustering

# CLI command: uvicorn app:app --reload --log-level debug
app = FastAPI(title="FATEC tool API",
              description="API to execute causal analysis and retrieve results from the FATEC tool.",
              version="1.0.0",
              contact={
                  "name": "Carlos Silva",
                  "email": "carlos.silva@inesctec.pt",
              })

# PostgreSQL connection
DATABASE_URL = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"\
               f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_NAME}"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# -- SQLAlchemy models --


class Task(Base):
    __tablename__ = "tasks"
    id = Column(String, primary_key=True)
    status = Column(String, default="pending")
    result = Column(JSONB, nullable=True)
    parameters = Column(JSONB)
    launch_time = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ClusteringTask(Base):
    __tablename__ = "clustering_tasks"
    id = Column(String, primary_key=True)
    status = Column(String, default="pending")
    result = Column(JSONB, nullable=True)
    parameters = Column(JSONB)
    launch_time = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Create table if it does not exist
Base.metadata.create_all(bind=engine)

# -- PyDantic models --


class TaskRequest(BaseModel):
    partner: str = Field(..., min_length=1, example="coopernico", description="The name of the partner.")
    consumers: List[str] = Field(..., min_items=1, example=["test", "test1", "test2"], description="List of consumers.")
    city: str = Field(..., min_length=1, example="Porto", description="The city where the consumers are located.")
    aggregated: int = Field(..., ge=0, le=1, example=1, description="Whether to aggregate consumers (0 = No, 1 = Yes).")


class TaskResponse(BaseModel):
    task_id: UUID
    launch_time: str
    status: str
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]]
    last_updated: str


class ClusteringRequest(BaseModel):
    task_id: UUID = Field(..., description="The ID of the previously completed task whose output will be clustered.")
    min_clusters: int = Field(..., ge=2, example=2, description="Minimum number of clusters to test.")
    max_clusters: int = Field(..., gt=2, example=11, description="Maximum number of clusters to test.")


class ClusteringResponse(BaseModel):
    clustering_id: UUID
    launch_time: str
    status: str
    inputs: Dict[str, Any]
    outputs: Optional[Dict[str, Any]]
    last_updated: str

# Tasks to execute asynchronously


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


def execute_clustering_task(clustering_id: str, clustering_request: ClusteringRequest):
    db = SessionLocal()
    try:
        # Prepare folder
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        directory_name = f"results/clustering_{timestamp}_{clustering_id}"
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
        logger.add(f"{directory_name}/out.log", level="DEBUG", colorize=False, backtrace=True, diagnose=True)

        inputs = {
            "task_id": str(clustering_request.task_id),
            "min_clusters": clustering_request.min_clusters,
            "max_clusters": clustering_request.max_clusters,
        }
        json.dump(inputs, open(f"{directory_name}/inputs.json", 'w'), indent=4)

        # Run clustering process
        result_data = run_clustering(
            str(clustering_request.task_id),
            directory_name,
            min_clusters=clustering_request.min_clusters,
            max_clusters=clustering_request.max_clusters
        )
        logger.debug("Clustering execution completed.")

        # Update DB
        task = db.query(ClusteringTask).filter(ClusteringTask.id == clustering_id).first()
        if task:
            task.status = "completed"
            task.result = result_data
            task.last_updated = datetime.utcnow()
            db.commit()

    except Exception as e:
        logger.exception("Error in clustering execution")
        task = db.query(ClusteringTask).filter(ClusteringTask.id == clustering_id).first()
        if task:
            task.status = "failed"
            task.result = {"error": str(e)}
            task.last_updated = datetime.utcnow()
            db.commit()

    finally:
        db.close()


# API Endpoints


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


@app.post("/clustering")
def clustering(clustering_request: ClusteringRequest, background_tasks: BackgroundTasks):
    try:
        clustering_id = str(uuid.uuid4())

        db = SessionLocal()

        parameters_json = clustering_request.dict()
        parameters_json['task_id'] = str(parameters_json['task_id'])

        new_task = ClusteringTask(
            id=clustering_id,
            status="pending",
            result=None,
            parameters=parameters_json,
            launch_time=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        db.add(new_task)
        db.commit()

        # Launch clustering asynchronously
        background_tasks.add_task(execute_clustering_task, clustering_id, clustering_request)

        return {"clustering_id": clustering_id, "status": "pending"}

    except Exception as e:
        logger.error(f"Unexpected error in clustering endpoint: {e}")
        return JSONResponse(status_code=500, content={"error": "Internal Server Error"})

    finally:
        db.close()


@app.get("/clustering_result/{clustering_id}", response_model=ClusteringResponse)
def get_clustering_result(clustering_id: str):
    db = SessionLocal()
    try:
        try:
            clustering_uuid = UUID(clustering_id, version=4)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid clustering_id format")

        task = db.query(ClusteringTask).filter(ClusteringTask.id == str(clustering_uuid)).first()
        if not task:
            raise HTTPException(status_code=404, detail="Clustering task not found")

        inputs = task.parameters if isinstance(task.parameters, dict) else json.loads(task.parameters)
        outputs = task.result if isinstance(task.result, dict) else json.loads(task.result) if task.result else None

        response_data = {
            "clustering_id": task.id,
            "launch_time": task.launch_time.isoformat(),
            "status": task.status,
            "inputs": inputs,
            "outputs": outputs,
            "last_updated": task.last_updated.isoformat(),
        }

        logger.debug(f"Returning clustering response: {response_data}")
        return response_data

    finally:
        db.close()