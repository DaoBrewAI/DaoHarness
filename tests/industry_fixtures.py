"""Two industry workflows built from the same 9-step skeleton. Test-only: industry vocabulary is allowed here."""
from __future__ import annotations


def _step(id, role, instructions, primitive, inputs, config=None, approval=None):
    return {"id": id, "role": role, "instructions": instructions, "primitive": primitive,
            "config": {} if config is None else config, "inputs": list(inputs), "approval": approval}


def _item(id, label, description):
    return {"id": id, "label": label, "description": description}


ACCOUNTING = {
    "description": (
        "Every time a new client signs with our firm we run the same intake. First we gather the documents they "
        "send and their business profile. Then we pull out the legal name, entity type and owner. Next we check "
        "the documents against our intake checklist and decide whether the packet is complete. After that we "
        "draft an email asking for anything missing, and the engagement partner approves it before it goes out. "
        "Finally we write an intake summary and file the client intake package."),
    "spec": {
        "domain": "Accounting client intake",
        "purpose": "Turn a new client's documents into a checked intake file and an approved request for missing items.",
        "input_requirements": [
            {"name": "documents", "description": "Text of the documents the client sent", "required": True},
            {"name": "profile", "description": "Short business profile of the client", "required": True},
        ],
        "deliverables": ["Client intake file", "Approved request for missing items"],
        "completion_criteria": ["Every checklist item is marked present, missing or unclear",
                                "The request for missing items is approved"],
        "steps": [
            _step("collect", "Intake coordinator", "Gather the client documents and profile.", "collect_inputs",
                  ["input:documents", "input:profile"]),
            _step("extract", "Intake specialist", "Extract the core entity facts.", "extract_fields",
                  ["step:collect"], {"fields": [
                      {"name": "legal_name", "description": "Registered legal name of the business", "required": True},
                      {"name": "entity_type", "description": "Entity type such as LLC or corporation", "required": True},
                      {"name": "owner_name", "description": "Name of the owner", "required": True}]}),
            _step("assess", "Intake specialist", "Check the documents against the intake checklist.",
                  "checklist_match", ["input:documents"], {"items": [
                      _item("prior_year_return", "Prior-year return", "Last year's filed tax return"),
                      _item("bank_statements", "Bank statements", "Recent bank statements for all accounts"),
                      _item("photo_id", "Photo ID", "Government photo ID of the owner"),
                      _item("engagement_letter", "Engagement letter", "Signed engagement letter"),
                      _item("payroll_reports", "Payroll reports", "Quarterly payroll reports"),
                      _item("prior_year_financials", "Prior-year financials", "Last year's financial statements")]}),
            _step("route", "Intake specialist", "Decide whether the intake packet is complete.", "classify_route",
                  ["step:assess"], {"categories": [
                      _item("complete_packet", "Complete packet", "Every checklist item is present"),
                      _item("missing_items", "Missing items", "One or more checklist items are missing or unclear")]}),
            _step("questions", "Intake specialist", "Ask what is needed to close the gaps.", "generate_questions",
                  ["step:assess", "step:extract"],
                  {"max_questions": 5, "focus": "Missing or unclear intake documents"}),
            _step("outreach", "Client service associate", "Draft an email to the client requesting missing items.",
                  "draft_message", ["step:assess", "step:questions"],
                  {"channel": "email", "audience": "client", "external": True}),
            _step("approval", "Engagement partner", "Review the outbound email before it is sent.", "human_approval",
                  ["step:outreach"], approval={"approver_role": "Engagement partner",
                                               "what_is_approved": "Outbound email"}),
            _step("summary", "Intake specialist", "Summarize the intake status.", "summarize_report",
                  ["step:extract", "step:assess", "step:questions"],
                  {"sections": ["Entity facts", "Checklist status", "Open questions"]}),
            _step("package", "Intake coordinator", "Assemble the intake file.", "deliver_package",
                  ["step:summary", "step:approval"], {"title": "Client intake file"}),
        ],
    },
    "sample": {
        "documents": ("Prior-year return for Northwind LLC attached. Bank statements for January through March. "
                      "Photo ID of the owner Dana Reyes. Employer number 12-3456789."),
        "profile": "Northwind LLC, single-member company owned by Dana Reyes.",
    },
}


RECRUITING = {
    "description": (
        "Whenever a new applicant comes in for an open role we follow our screening process. First we gather the "
        "resume and the role brief. Then we extract the candidate's name, years of experience and current title. "
        "Next we score the resume against the role criteria and decide to advance, hold or decline. After that we "
        "draft an email to the candidate, and the hiring manager approves it. Finally we write a screening summary "
        "and assemble the screening packet."),
    "spec": {
        "domain": "Recruiting candidate screening",
        "purpose": "Turn a candidate's resume into a scored screening packet and an approved candidate email.",
        "input_requirements": [
            {"name": "resume", "description": "Text of the candidate's resume", "required": True},
            {"name": "role_brief", "description": "Short brief of the open role", "required": True},
        ],
        "deliverables": ["Screening packet", "Approved candidate email"],
        "completion_criteria": ["Every criterion is scored with evidence",
                                "The candidate email is approved"],
        "steps": [
            _step("collect", "Recruiting coordinator", "Gather the resume and role brief.", "collect_inputs",
                  ["input:resume", "input:role_brief"]),
            _step("extract", "Sourcer", "Extract the candidate's core facts.", "extract_fields",
                  ["step:collect"], {"fields": [
                      {"name": "full_name", "description": "Candidate's full name", "required": True},
                      {"name": "years_experience", "description": "Total years of experience", "required": True},
                      {"name": "current_title", "description": "Most recent job title", "required": True}]}),
            _step("assess", "Recruiter", "Score the resume against the role criteria.", "screen_score",
                  ["input:resume", "input:role_brief"], {"scale_max": 5, "criteria": [
                      {**_item("python", "Python", "Production Python experience"), "weight": 3},
                      {**_item("sql", "SQL", "Strong SQL and data modeling"), "weight": 3},
                      {**_item("leadership", "Leadership", "Has led an engineering team"), "weight": 2},
                      {**_item("pipelines", "Pipelines", "Built and operated data pipelines"), "weight": 2}]}),
            _step("route", "Recruiter", "Decide the next step for the candidate.", "classify_route",
                  ["step:assess"], {"categories": [
                      _item("advance", "Advance", "Strong match; move to interviews"),
                      _item("hold", "Hold", "Partial match; keep warm"),
                      _item("decline", "Decline", "Not a match for this role")]}),
            _step("questions", "Recruiter", "Ask what is needed to close the gaps.", "generate_questions",
                  ["step:assess", "step:extract"],
                  {"max_questions": 5, "focus": "Gaps between the resume and the role criteria"}),
            _step("outreach", "Recruiter", "Draft an email to the candidate about next steps.", "draft_message",
                  ["step:assess", "step:questions"],
                  {"channel": "email", "audience": "candidate", "external": True}),
            _step("approval", "Hiring manager", "Review the outbound email before it is sent.", "human_approval",
                  ["step:outreach"], approval={"approver_role": "Hiring manager",
                                               "what_is_approved": "Outbound email"}),
            _step("summary", "Recruiter", "Summarize the screening result.", "summarize_report",
                  ["step:extract", "step:assess", "step:questions"],
                  {"sections": ["Candidate facts", "Criteria scores", "Open questions"]}),
            _step("package", "Recruiting coordinator", "Assemble the screening packet.", "deliver_package",
                  ["step:summary", "step:approval"], {"title": "Screening packet"}),
        ],
    },
    "sample": {
        "resume": ("Priya Shah, senior data engineer with 7 years of experience building pipelines in Python and "
                   "SQL. Led a team of four."),
        "role_brief": "Senior data engineer, remote, Python and SQL, team lead experience preferred.",
    },
}


# --- per-item fan-out: one packet holds several applicants --------------------------------------------------------

def recruiting_fan_out_spec():
    """split the packet into applicants, run four steps once per applicant, approve every draft in one
    batch, then summarise across all applicants. 8 of 9 steps are automatic."""
    criteria = RECRUITING["spec"]["steps"][2]["config"]
    categories = RECRUITING["spec"]["steps"][3]["config"]
    fields = RECRUITING["spec"]["steps"][1]["config"]
    return {
        "domain": "Recruiting applicant screening",
        "purpose": "Screen every applicant in a packet and approve one email per applicant.",
        "input_requirements": [
            {"name": "packet", "description": "Text of several applicants' resumes", "required": True},
            {"name": "role_brief", "description": "Short brief of the open role", "required": True},
        ],
        "deliverables": ["Screening packet", "Approved applicant emails"],
        "completion_criteria": ["Every applicant is scored with evidence", "Every applicant email is approved"],
        "steps": [
            _step("collect", "Recruiting coordinator", "Gather the packet and role brief.", "collect_inputs",
                  ["input:packet", "input:role_brief"]),
            _step("split", "Recruiting coordinator", "Split the packet into one entry per applicant.", "split_items",
                  ["input:packet"], {"item_noun": "applicant", "max_items": 20}),
            {**_step("extract", "Sourcer", "Extract the applicant's core facts.", "extract_fields",
                     ["step:split"], fields), "for_each": "split"},
            {**_step("assess", "Recruiter", "Score the applicant against the role criteria.", "screen_score",
                     ["step:split", "input:role_brief"], criteria), "for_each": "split"},
            {**_step("route", "Recruiter", "Decide the next step for the applicant.", "classify_route",
                     ["step:assess"], categories), "for_each": "split"},
            {**_step("outreach", "Recruiter", "Draft an email to the applicant about next steps.", "draft_message",
                     ["step:extract", "step:route"],
                     {"channel": "email", "audience": "applicant", "external": True}), "for_each": "split"},
            _step("approval", "Hiring manager", "Review every outbound email before it is sent.", "human_approval",
                  ["step:outreach"], approval={"approver_role": "Hiring manager",
                                               "what_is_approved": "Outbound applicant emails"}),
            _step("summary", "Recruiter", "Rank the applicants and summarise the screening.", "summarize_report",
                  ["step:assess", "step:route", "step:approval"],
                  {"sections": ["Ranking", "Per-applicant notes"]}),
            _step("package", "Recruiting coordinator", "Assemble the screening packet.", "deliver_package",
                  ["step:summary", "step:approval"], {"title": "Screening packet"}),
        ],
    }


FAN_OUT_APPLICANTS = (
    ("Priya Shah", "7 years", "Senior Data Engineer", "Built pipelines in Python and SQL. Led a team of four."),
    ("Marco Silva", "3 years", "Data Analyst", "Strong SQL and dashboards; some Python scripting."),
    ("Li Wei", "12 years", "Engineering Manager", "Led data platform teams; Python and Spark pipelines."),
)


def recruiting_fan_out_sample():
    """Three applicants with different names, years and titles, plus the shared role brief."""
    blocks = [f"Applicant: {name}\nExperience: {years}\nCurrent title: {title}\n{notes}"
              for name, years, title, notes in FAN_OUT_APPLICANTS]
    return {"packet": "Applicant packet for the senior data engineer role.\n\n" + "\n\n".join(blocks),
            "role_brief": RECRUITING["sample"]["role_brief"]}


def _line_value(text, label):
    for line in text.splitlines():
        if line.startswith(label + ": "):
            return line[len(label) + 2:].strip(), line.strip()
    return "", ""


def fan_out_fake_outputs():
    """FakeModel step_overrides that answer only from the text they receive, so a value taken from another
    applicant fails that applicant's citations check."""
    def split(input_data):
        packet = input_data["inputs"]["input:packet"]
        return {"items": [{"label": line.split(": ", 1)[1], "start_quote": line}
                          for line in packet.splitlines() if line.startswith("Applicant: ")]}

    def extract(input_data):
        text = input_data["inputs"]["step:split"]["text"]
        rows = []
        for name, label in (("full_name", "Applicant"), ("years_experience", "Experience"),
                            ("current_title", "Current title")):
            value, quote = _line_value(text, label)
            rows.append({"name": name, "found": bool(value), "value": value, "source_quotes": [quote] if quote else []})
        return {"fields": rows}

    def outreach(input_data):
        fields = {f["name"]: f["value"] for f in input_data["inputs"]["step:extract"]["fields"]}
        return {"subject": f"Next steps for {fields.get('full_name', '')}",
                "body": f"Hello {fields.get('full_name', '')}, thank you for applying.", "notes": ""}

    return {"split": split, "extract": extract, "outreach": outreach}
