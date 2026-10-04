from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, Response

from app.schemas.task import (
    ApprovalDecisionRequest,
    HistoryResponse,
    TaskCreateRequest,
    TaskResponse,
)
from app.services.task_service import InvalidTaskState, TaskNotFound, TaskService

router = APIRouter(prefix="/tasks", tags=["tasks"])


def get_service(request: Request) -> TaskService:
    return request.app.state.task_service


@router.post("", response_model=TaskResponse, status_code=202, summary="Submit a natural-language task")
def create_task(
    body: TaskCreateRequest,
    request: Request,
    response: Response,
    background: BackgroundTasks,
    wait: bool = Query(False, description="Block until the run finishes or pauses for approval"),
    service: TaskService = Depends(get_service),
) -> TaskResponse:
    if body.simulate_fault and not request.app.state.settings.enable_fault_injection:
        raise HTTPException(status_code=403, detail="Fault injection is disabled.")
    task_id = service.create_task(body.task, body.simulate_fault)
    if wait:
        response.status_code = 200
        return service.run_task(task_id)
    background.add_task(service.run_task, task_id)
    return service.get_task(task_id)


@router.get("/{task_id}", response_model=TaskResponse, summary="Task status, plan, result and evidence")
def get_task(task_id: str, service: TaskService = Depends(get_service)) -> TaskResponse:
    try:
        return service.get_task(task_id)
    except TaskNotFound:
        raise HTTPException(status_code=404, detail="Task not found")


@router.get("/{task_id}/history", response_model=HistoryResponse, summary="Structured execution history")
def get_history(task_id: str, service: TaskService = Depends(get_service)) -> HistoryResponse:
    try:
        return service.get_history(task_id)
    except TaskNotFound:
        raise HTTPException(status_code=404, detail="Task not found")


def _decide(
    task_id: str,
    approved: bool,
    body: ApprovalDecisionRequest | None,
    response: Response,
    background: BackgroundTasks,
    wait: bool,
    service: TaskService,
) -> TaskResponse:
    note = body.note if body else ""
    try:
        service.decide_approval(task_id, approved, note)
    except TaskNotFound:
        raise HTTPException(status_code=404, detail="Task not found")
    except InvalidTaskState as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    if wait:
        response.status_code = 200
        return service.run_task(task_id)
    background.add_task(service.run_task, task_id)
    return service.get_task(task_id)


@router.post("/{task_id}/approve", response_model=TaskResponse, status_code=202, summary="Approve the pending action")
def approve(
    task_id: str,
    response: Response,
    background: BackgroundTasks,
    body: ApprovalDecisionRequest | None = None,
    wait: bool = Query(False),
    service: TaskService = Depends(get_service),
) -> TaskResponse:
    return _decide(task_id, True, body, response, background, wait, service)


@router.post("/{task_id}/reject", response_model=TaskResponse, status_code=202, summary="Reject the pending action")
def reject(
    task_id: str,
    response: Response,
    background: BackgroundTasks,
    body: ApprovalDecisionRequest | None = None,
    wait: bool = Query(False),
    service: TaskService = Depends(get_service),
) -> TaskResponse:
    return _decide(task_id, False, body, response, background, wait, service)
