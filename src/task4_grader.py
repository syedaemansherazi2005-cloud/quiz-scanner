"""
Task 4 — Quiz Grading [20 points]
Compares student answers against answer key and produces a GradeReport.
"""

from dataclasses import dataclass, field
from typing import Optional
from task1_qr_decoder import AnswerKey
from task3_bubble_reader import StudentAnswers


@dataclass
class QuestionResult:
    question: str
    student_answer: Optional[str]   # None = unattempted, "INVALID" = multi-filled
    correct_answer: str
    status: str                      # "correct" | "incorrect" | "unattempted" | "invalid"
    marks: float = 0.0

    def symbol(self) -> str:
        return {"correct": "✓", "incorrect": "✗", "unattempted": "—", "invalid": "⚠"}.get(self.status, "?")

    def to_dict(self):
        return {
            "question": self.question,
            "student_answer": self.student_answer,
            "correct_answer": self.correct_answer,
            "status": self.status,
            "marks": self.marks,
            "symbol": self.symbol(),
        }


@dataclass
class GradeReport:
    quiz_set: str = ""
    student_name: str = ""
    reg_no: str = ""
    class_name: str = ""
    subject: str = ""

    part1_results: list = field(default_factory=list)
    part2_results: list = field(default_factory=list)

    correct: int = 0
    incorrect: int = 0
    unattempted: int = 0
    invalid: int = 0

    total_questions: int = 0
    total_marks: float = 0.0
    max_marks: float = 0.0
    percentage: float = 0.0
    grade: str = ""
    negative_marking: float = 0.0

    def to_dict(self):
        return {
            "quiz_set": self.quiz_set,
            "student_name": self.student_name,
            "reg_no": self.reg_no,
            "class": self.class_name,
            "subject": self.subject,
            "part1": [r.to_dict() for r in self.part1_results],
            "part2": [r.to_dict() for r in self.part2_results],
            "summary": {
                "correct": self.correct,
                "incorrect": self.incorrect,
                "unattempted": self.unattempted,
                "invalid": self.invalid,
                "total_questions": self.total_questions,
                "total_marks": self.total_marks,
                "max_marks": self.max_marks,
                "percentage": round(self.percentage, 2),
                "grade": self.grade,
                "negative_marking": self.negative_marking,
            },
        }


def letter_grade(percentage: float) -> str:
    """Standard letter grade scale."""
    if percentage >= 90:
        return "A+"
    elif percentage >= 85:
        return "A"
    elif percentage >= 80:
        return "A-"
    elif percentage >= 75:
        return "B+"
    elif percentage >= 70:
        return "B"
    elif percentage >= 65:
        return "B-"
    elif percentage >= 60:
        return "C+"
    elif percentage >= 55:
        return "C"
    elif percentage >= 50:
        return "C-"
    elif percentage >= 45:
        return "D"
    else:
        return "F"


def grade_part(student_part: dict, key_part: dict,
               negative_marking: float = 0.0) -> tuple[list[QuestionResult], float]:
    """Grade one part (Part-I or Part-II). Returns (results, marks_earned)."""
    results = []
    marks = 0.0

    # Ensure we iterate through all questions in the key
    for q_key in sorted(key_part.keys()):
        correct_ans = key_part[q_key]
        student_ans = student_part.get(q_key)  # None if missing

        if student_ans is None:
            status = "unattempted"
            q_marks = 0.0
        elif student_ans == "INVALID":
            status = "invalid"
            q_marks = 0.0
        elif student_ans == correct_ans:
            status = "correct"
            q_marks = 1.0
        else:
            status = "incorrect"
            q_marks = -negative_marking if negative_marking > 0 else 0.0

        marks += q_marks
        results.append(QuestionResult(
            question=q_key,
            student_answer=student_ans,
            correct_answer=correct_ans,
            status=status,
            marks=q_marks,
        ))

    return results, marks


def grade_quiz(student_answers: StudentAnswers, answer_key: AnswerKey,
               student_name: str = "", reg_no: str = "") -> GradeReport:
    """
    Main function: Task 4 entry point.
    Accepts StudentAnswers and AnswerKey, returns GradeReport.
    """
    report = GradeReport(
        quiz_set=answer_key.quiz_set,
        student_name=student_name,
        reg_no=reg_no,
        class_name=answer_key.class_name,
        subject=answer_key.subject,
        negative_marking=answer_key.negative_marking,
    )

    neg = answer_key.negative_marking

    p1_results, p1_marks = grade_part(student_answers.part1, answer_key.part1, neg)
    p2_results, p2_marks = grade_part(student_answers.part2, answer_key.part2, neg)

    report.part1_results = p1_results
    report.part2_results = p2_results

    all_results = p1_results + p2_results
    report.correct     = sum(1 for r in all_results if r.status == "correct")
    report.incorrect   = sum(1 for r in all_results if r.status == "incorrect")
    report.unattempted = sum(1 for r in all_results if r.status == "unattempted")
    report.invalid     = sum(1 for r in all_results if r.status == "invalid")

    report.total_questions = len(all_results)
    report.max_marks = float(report.total_questions)
    report.total_marks = max(0.0, p1_marks + p2_marks)  # clamp to 0
    report.percentage = (report.total_marks / report.max_marks * 100) if report.max_marks > 0 else 0.0
    report.grade = letter_grade(report.percentage)

    print(f"[Task 4] Graded: {report.correct}/{report.total_questions} correct | "
          f"Score: {report.total_marks:.1f}/{report.max_marks:.0f} | "
          f"Grade: {report.grade} ({report.percentage:.1f}%)")
    return report


def print_report(report: GradeReport):
    """Pretty-print the grade report to console."""
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  QUIZ GRADE REPORT")
    print(sep)
    print(f"  Student : {report.student_name or 'Unknown'}")
    print(f"  Reg No  : {report.reg_no or 'Unknown'}")
    print(f"  Set     : {report.quiz_set}   |   {report.subject}")
    print(sep)

    for label, results in [("Part-I", report.part1_results), ("Part-II", report.part2_results)]:
        print(f"\n  {label}")
        print("  " + "-" * 40)
        print(f"  {'Q':<6} {'Student':>8} {'Key':>5} {'':>4} {'Marks':>6}")
        print("  " + "-" * 40)
        for r in results:
            print(f"  {r.question:<6} {str(r.student_answer or '—'):>8} "
                  f"{r.correct_answer:>5} {r.symbol():>4} {r.marks:>6.2f}")

    print(f"\n{sep}")
    print(f"  Correct     : {report.correct}")
    print(f"  Incorrect   : {report.incorrect}")
    print(f"  Unattempted : {report.unattempted}")
    if report.invalid:
        print(f"  Invalid     : {report.invalid}  ⚠")
    print(f"  Score       : {report.total_marks:.1f} / {report.max_marks:.0f}")
    print(f"  Percentage  : {report.percentage:.1f}%")
    print(f"  Grade       : {report.grade}")
    if report.negative_marking:
        print(f"  Neg. Marking: -{report.negative_marking} per wrong")
    print(sep)


# ── Demo ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from task1_qr_decoder import AnswerKey
    from task3_bubble_reader import StudentAnswers

    # Synthetic demo
    key = AnswerKey(
        quiz_set="C",
        part1={f"Q{i:02d}": c for i, c in enumerate(["D","A","B","A","D","A","A","B"], 1)},
        part2={f"Q{i:02d}": c for i, c in enumerate(["C","D","D","D","C","C","C","B"], 1)},
    )
    student = StudentAnswers(
        part1={f"Q{i:02d}": c for i, c in enumerate(["D","A","C","A","D","B","A","B"], 1)},
        part2={f"Q{i:02d}": c for i, c in enumerate(["C","D","D","A","C","C","C","B"], 1)},
    )
    report = grade_quiz(student, key, student_name="Ali Hassan", reg_no="FA22-BSE-041")
    print_report(report)
