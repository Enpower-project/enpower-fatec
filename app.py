import os
import json
import time
import uuid
from loguru import logger
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import BackgroundTasks
from sqlalchemy import DateTime, func
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI, HTTPException
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import create_engine, Column, String
from sqlalchemy.ext.declarative import declarative_base

from main import run_consumers

load_dotenv(".env")
app = FastAPI()

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


# Input schema matching your expected JSON format
class TaskRequest(BaseModel):
    partner: str
    consumers: list[str]
    city: str
    aggregated: int


# API Endpoint to execute task in background
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


# API Endpoint to create and execute a task
@app.post("/execute")
def execute(task_request: TaskRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())

    db = SessionLocal()
    try:
        # 1️⃣ Store "pending" task in DB
        new_task = Task(
            id=task_id,
            status="pending",
            result=None,
            parameters=json.dumps(task_request.dict()),
            launch_time=datetime.utcnow(),
            last_updated=datetime.utcnow()
        )
        db.add(new_task)
        db.commit()

        # 2️⃣ Run the tool asynchronously
        background_tasks.add_task(execute_task, task_id, task_request)

        return {"task_id": task_id, "status": "pending"}

    finally:
        db.close()


# API Endpoint to check task result
@app.get("/result/{task_id}")
def get_result(task_id: str):
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()
    db.close()

    if not task:
        raise HTTPException(status_code=404, detail="Task ID not found")

    if task.status == "pending":
        return {"task_id": task.id,
                "launch_time": task.launch_time,
                "status": "pending"}
    else:
        return {
            "task_id": task.id,
            "launch_time": task.launch_time,
            "status": task.status,
            "inputs": json.loads(task.parameters),
            "outputs": task.result,
            "last_updated": task.last_updated,
        }
