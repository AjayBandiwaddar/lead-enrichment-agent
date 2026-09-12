from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    excerpt: str = Field(min_length=1)


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str = Field(min_length=1)
    evidence: list[EvidenceRef] = Field(min_length=1)


class ContactEmail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3)
    email_type: str = Field(min_length=1)
    evidence: list[EvidenceRef] = Field(min_length=1)


class LeadershipPerson(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    linkedin_url: str | None = None
    evidence: list[EvidenceRef] = Field(min_length=1)


class OverviewICPExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overview: Claim | None = None
    icp_summary: Claim | None = None
    target_segments: list[str] = Field(default_factory=list)
    buyer_roles: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)


class ContactsExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contacts: list[ContactEmail] = Field(default_factory=list)


class LeadershipExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    leadership: list[LeadershipPerson] = Field(default_factory=list)


class ContactsLeadershipExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    contacts: list[ContactEmail] = Field(default_factory=list)
    leadership: list[LeadershipPerson] = Field(default_factory=list)


class CompanyExtraction(BaseModel):
    """Final extraction contract before deterministic verification."""

    model_config = ConfigDict(extra="forbid")

    overview: Claim | None = None
    icp_summary: Claim | None = None
    target_segments: list[str] = Field(default_factory=list)
    buyer_roles: list[str] = Field(default_factory=list)
    use_cases: list[str] = Field(default_factory=list)
    contacts: list[ContactEmail] = Field(default_factory=list)
    leadership: list[LeadershipPerson] = Field(default_factory=list)