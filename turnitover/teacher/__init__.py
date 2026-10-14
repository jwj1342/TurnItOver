"""Visible-information teachers for auditable interactive episodes."""

from turnitover.teacher.providers import KimiTeacher, LocalQwenTeacher, MockTeacher, Teacher

__all__ = ["Teacher", "MockTeacher", "LocalQwenTeacher", "KimiTeacher"]
