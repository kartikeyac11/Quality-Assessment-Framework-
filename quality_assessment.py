import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
import pytz
import pandas as pd
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ML models
from sentence_transformers import CrossEncoder
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
status_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)


PRIORITY_ORDER = ["Highest", "High", "Medium", "Low", "Lowest"]
STATUS_EQUIVALENTS = {
    "TODO": "TODO",
    "COMPLETED": "COMPLETED",
    "IN_PROGRESS": "IN_PROGRESS",
    "INPROGRESS": "IN_PROGRESS",
    "BLOCKED": "BLOCKED",
    "ARCHIVED": "ARCHIVED",
    "DONE": "COMPLETED",
    "FINISHED": "COMPLETED",
    "DOING": "IN_PROGRESS",
    "IN PROGRESS": "IN_PROGRESS"
}

WEIGHTS = {
    "title_description": 0.30,
    "status": 0.15,
    "remarks": 0.20,
    "assignee": 0.15,
    "due_date": 0.10,
    "priority": 0.10,
}

HALLUCINATION_SIM_THRESHOLD = 0.65
IST_TZ = pytz.timezone("Asia/Kolkata")

@dataclass
class TaskGT:
    taskId: str
    title: str
    description: str
    remarks: str
    assignees: List[str]
    dueDate: Optional[str]
    status: str
    priority: Optional[str]
    project: Optional[str]
    raw: Dict[str, Any]

@dataclass
class TaskSys:
    id: str
    title: str
    description: str
    remarks: str
    assignee: Optional[str]
    dueDate: Optional[str]
    status: str
    priority: Optional[str]
    projectId: Optional[str]
    raw: Dict[str, Any]

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_status(s: Optional[str]) -> str:
    if not s:
        return "TODO"
    s = s.strip().upper()
    return STATUS_EQUIVALENTS.get(s, s)

def normalize_priority(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    p = str(p).strip()
    if p.lower() in ("null", ""):
        return None
    for known in PRIORITY_ORDER:
        if p.lower() == known.lower():
            return known
    return p.capitalize()

def parse_and_normalize_date_to_ist(datestr: Optional[str]) -> Optional[datetime]:
    if not datestr:
        return None
    try:
        from dateutil import parser as dateparser
        dt = dateparser.parse(datestr)
        if dt.tzinfo is None:
            dt = pytz.UTC.localize(dt)
        return dt.astimezone(IST_TZ)
    except Exception:
        return None

def priority_score(sys_p, gt_p):
    sys_p = normalize_priority(sys_p); gt_p = normalize_priority(gt_p)
    if sys_p == gt_p:
        return 1.0
    if sys_p not in PRIORITY_ORDER or gt_p not in PRIORITY_ORDER:
        return 0.0
    sys_idx, gt_idx = PRIORITY_ORDER.index(sys_p), PRIORITY_ORDER.index(gt_p)
    dist = abs(sys_idx - gt_idx)
    return max(0.0, 1.0 - 0.5*dist)

def status_score(sys_status: str, gt_status: str) -> float:
    sys_status = normalize_status(sys_status)
    gt_status = normalize_status(gt_status)

    if sys_status == gt_status:
        return 1.0

    emb1 = status_model.encode(sys_status, convert_to_tensor=True)
    emb2 = status_model.encode(gt_status, convert_to_tensor=True)
    sim = float(util.pytorch_cos_sim(emb1, emb2).item())
    return sim

def assignee_score(sys_assignee: Optional[str], gt_assignees: List[str]) -> float:
    if not sys_assignee:
        return 0.0
    if len(gt_assignees) > 1:
        return (1/(len(gt_assignees))) if sys_assignee.strip().lower() in [a.strip().lower() for a in gt_assignees] else 0.0
    else:
        return 1.0 if sys_assignee.strip().lower() in [a.strip().lower() for a in gt_assignees] else 0.0
    

def due_date_score(sys_due: Optional[datetime], gt_due: Optional[datetime]) -> (float, bool):
    if sys_due is None and gt_due is None:
        return 0.5, False   # both missing → neutral

    if sys_due is None or gt_due is None:
        return 0.0, False   # one missing → mismatch
    same_day = (sys_due.date() == gt_due.date())
    if same_day:
        delta_hours = abs((sys_due - gt_due).total_seconds()) / 3600.0
        if delta_hours < 1e-3:
            return 1.0, True
        elif delta_hours <= 12:
            return 0.6, True
    return 0.0, False



class QualityAssessor:
    def __init__(self, gt_json: Dict[str, Any], ws_json: Dict[str, Any]):
        print("Loading models...")
        self.cross_encoder = CrossEncoder(
        "cross-encoder/stsb-roberta-large", 
        device=0 if DEVICE == "cuda" else "cpu"
        )
        self.nli = pipeline(
        "text-classification",
        model="facebook/bart-large-mnli",
        return_all_scores=True,
        device=0 if DEVICE == "cuda" else -1
        )
        self.actionability_classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=0 if DEVICE == "cuda" else -1
        )
        print(f"Models loaded on {DEVICE.upper()}.")

        self.gt_tasks: List[TaskGT] = self._parse_gt_tasks(gt_json)
        self.sys_tasks: List[TaskSys] = self._parse_sys_tasks(ws_json)
        self.project_meta = {}
        if "projectMeta" in ws_json:
            for proj in ws_json["projectMeta"]:
                self.project_meta[proj.get("id")] = proj


    def _parse_gt_tasks(self, gt_json: Dict[str, Any]) -> List[TaskGT]:
        return [TaskGT(
            taskId=t.get("taskId"),
            title=t.get("taskTitle", ""),
            description=t.get("taskDescription", ""),
            remarks=t.get("remarks", ""),
            assignees=t.get("assignees", []) or [],
            dueDate=t.get("dueDate"),
            status=t.get("status", ""),
            priority=t.get("priority"),
            project=t.get("project"),
            raw=t,
        ) for t in gt_json.get("tasks", [])]

    def _parse_sys_tasks(self, ws_json: Dict[str, Any]) -> List[TaskSys]:
        return [TaskSys(
            id=t.get("id"),
            title=t.get("title", ""),
            description=t.get("description", ""),
            remarks=t.get("remarks", ""),
            assignee=t.get("assignee"),
            dueDate=t.get("dueDate"),
            status=t.get("status", ""),
            priority=t.get("priority"),
            projectId=t.get("projectId"),
            raw=t,
        ) for t in ws_json.get("tasks", [])]

    def map_system_to_ground_truth(self):
        mappings, hallu = [], []
        for sys_task in self.sys_tasks:
            best_score, best_i = -1, None
            for i, gt_task in enumerate(self.gt_tasks):
                title_score = float(self.cross_encoder.predict([(sys_task.title, gt_task.title)])[0])
                desc_score = float(self.cross_encoder.predict([(sys_task.description, gt_task.description)])[0])
                combined_score = 0.6 * title_score + 0.4 * desc_score
                if combined_score > best_score:
                    best_score = combined_score
                    best_i = i
            gt_match = self.gt_tasks[best_i] if best_i is not None else None
            if best_score < HALLUCINATION_SIM_THRESHOLD:
                hallu.append(sys_task)
            mappings.append({"sys_task": sys_task, "gt_task": gt_match, "similarity": best_score})
        return mappings, hallu

    def detect_granularity(self, mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
        gt_to_sys = {}
        sys_to_gt_sim = {}

        # Build mapping structures
        for m in mappings:
            gt = m["gt_task"].taskId if m["gt_task"] else None
            sys = m["sys_task"].id
            sim = m["similarity"]

            if gt:
                gt_to_sys.setdefault(gt, []).append(sys)
            sys_to_gt_sim.setdefault(sys, []).append((gt, sim))

    # Rule 1: Over-granular → one GT maps to multiple SYS
        over = {gt: syss for gt, syss in gt_to_sys.items() if len(syss) > 1}

    # Rule 2: Under-granular → one SYS maps strongly (>0.68) to multiple GTs
        under = {}
        for sys, gtsims in sys_to_gt_sim.items():
            high_sim_gts = [gt for gt, sim in gtsims if gt and sim > 0.68]
            if len(high_sim_gts) > 1:
                under[sys] = high_sim_gts

        return {"over_granular": over, "under_granular": under}

    def evaluate_remarks(self, gt_remark: str, sys_remark: str) -> Dict[str, Any]:
        def nli_entailment_score(premise: str, hypothesis: str):
            if not premise or not hypothesis:
                return (0.0, 0.0, 0.0)
            res = self.nli({"text": premise, "text_pair": hypothesis}, truncation=True)
            if isinstance(res, list) and len(res) > 0:
                logits = res[0] if isinstance(res[0], list) else res
                scores = {d["label"].upper(): float(d["score"]) for d in logits}
                e = scores.get("ENTAILMENT", 0.0)
                c = scores.get("CONTRADICTION", 0.0)
                n = scores.get("NEUTRAL", 0.0)
                s = e + c + n
                if s > 0:
                    e, c, n = e/s, c/s, n/s
                return (e, c, n)
            return (0.0, 0.0, 0.0)

        e1, c1, n1 = nli_entailment_score(gt_remark, sys_remark)
        e2, c2, n2 = nli_entailment_score(sys_remark, gt_remark)
        remark_score = 0.0 
        if e1 > 0.5 and e2 > 0.5:
            flag = "Perfect"
            remark_score = max(e1, e2)
        elif max(c1, c2) > 0.5:
            flag = "Severe contradiction"
            remark_score = 1.0 - max(c1, c2)
        elif n1 > 0.5 and e2 > 0.5:
            flag = "Hallucination (SYS adds info)"
            remark_score = 1.0 - max(n1,e2)
        elif n2 > 0.5 and e1 > 0.5:
            flag = "Missing info"
            remark_score = 1.0 - max(n2,e1)
        else:
            flag = "Borderline"
            remark_score = 1.0 - max((e1+n1+c1)/3 , (e2+n2+c2)/3)
        return {"remark_score": remark_score, "flag": flag}

    def classify_hallucinated_task(self, task: TaskSys) -> str:
        result = self.actionability_classifier(task.title + " " + task.description, candidate_labels=["Actionable", "Informational"])
        top_label = result["labels"][0]
        return "Actionable" if "Actionable" in top_label else "Informational"
    def detect_wrong_project(self, mapping: Dict[str, Any]) -> bool:
        sys_task: TaskSys = mapping["sys_task"]
        gt_task: TaskGT = mapping["gt_task"]

        if gt_task is None:
            return False

    # Resolve project name from sys_task.projectId via project_meta
        sys_proj = None
        if sys_task.projectId and sys_task.projectId in self.project_meta:
            sys_proj = self.project_meta[sys_task.projectId].get("title")

    # Compare GT project (string) vs SYS project title
        if sys_proj and gt_task.project:
         return sys_proj.strip().lower() != gt_task.project.strip().lower()

        return False


    def assess(self) -> Dict[str, Any]:
        mappings, hallu = self.map_system_to_ground_truth()
        results = []
        for m in mappings:
            sys_task: TaskSys = m["sys_task"]
            gt_task: Optional[TaskGT] = m["gt_task"]
            similarity = m["similarity"]
            wrong_project_flag = self.detect_wrong_project(m)
            if similarity < HALLUCINATION_SIM_THRESHOLD:
                continue
            s_score = status_score(sys_task.status, gt_task.status if gt_task else "")
            a_score = assignee_score(sys_task.assignee, gt_task.assignees if gt_task else [])
            sys_due_dt = parse_and_normalize_date_to_ist(sys_task.dueDate)
            gt_due_dt = parse_and_normalize_date_to_ist(gt_task.dueDate if gt_task else None)
            d_score, due_match = due_date_score(sys_due_dt, gt_due_dt)
            p_score = priority_score(sys_task.priority, gt_task.priority if gt_task else None)
            remarks_eval = self.evaluate_remarks(gt_task.remarks if gt_task else "", sys_task.remarks)
            weighted = (WEIGHTS["title_description"] * similarity + WEIGHTS["status"] * s_score +
                        WEIGHTS["remarks"] * remarks_eval["remark_score"] + WEIGHTS["assignee"] * a_score +
                        WEIGHTS["due_date"] * d_score + WEIGHTS["priority"] * p_score)
            results.append({
                "sys_task_id": sys_task.id,
                "sys_title": sys_task.title,
                "matched_gt_id": gt_task.taskId if gt_task else None,
                "matched_gt_title": gt_task.title if gt_task else None,
                "similarity": similarity,
                "status_score": s_score,
                "assignee_score": a_score,
                "due_date_score": d_score,
                "priority_score": p_score,
                "remark_flag": remarks_eval["flag"],
                "wrong_project": wrong_project_flag,
                "overall_accuracy(%)": weighted*100
            })
        hallucinations = []
        for h in hallu:
            classification = self.classify_hallucinated_task(h)
            hallucinations.append({"sys_task_id": h.id, "sys_title": h.title, "classification": classification})
        return {"per_task": results, "hallucinations": hallucinations, "mappings": mappings}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Path to ground truth JSON file")
    parser.add_argument("--ws", required=True, help="Path to system workspace JSON file")
    parser.add_argument("--out", default="assessment_results.json", help="Output file base name")
    args = parser.parse_args()

    gt_json = load_json(args.gt)
    ws_json = load_json(args.ws)
    assessor = QualityAssessor(gt_json=gt_json, ws_json=ws_json)
    results = assessor.assess()
    matched_df = pd.DataFrame(results["per_task"])
    hallucinated_df = pd.DataFrame(results["hallucinations"])
    granularity = assessor.detect_granularity(results["mappings"])
    if not matched_df.empty:
        matched_df["granularity"] = matched_df.apply(
            lambda r: "Over-granular" if r["matched_gt_id"] in granularity["over_granular"]
            else "Under-granular" if r["sys_task_id"] in granularity["under_granular"]
            else "Aptly granular", axis=1)

        
        avg_accuracy = matched_df["overall_accuracy(%)"].mean()

        
        matched_gt_ids = set(matched_df["matched_gt_id"].dropna())
        total_gt = len(assessor.gt_tasks)
        recall = len(matched_gt_ids) / total_gt if total_gt else 0

        completeness = f"Completeness: {recall*100:.2f}%"

        
        matched_df.loc[-1] = ["" for _ in matched_df.columns]
        matched_df.loc[-1, "overall_accuracy(%)"] = f"Average: {avg_accuracy:.2f}% | {completeness}"
        matched_df.index = matched_df.index + 1  # shift index
        matched_df = matched_df.sort_index()



    
    matched_csv_path = args.out.replace(".json", "_matched.csv")
    hallucinated_csv_path = args.out.replace(".json", "_hallucinated.csv")
    matched_df.to_csv(matched_csv_path, index=False)
    hallucinated_df.to_csv(hallucinated_csv_path, index=False)
    print("CSV files saved to", matched_csv_path, "and", hallucinated_csv_path)

if __name__ == "__main__":
    main()
