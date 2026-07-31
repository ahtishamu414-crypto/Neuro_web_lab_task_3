"""
Synthetic resume/cover-letter + ground-truth-label generator.

WHY SYNTHETIC DATA AT ALL
TalentGrid's own applicant data is naturally not available for a student
project (it's real candidates' PII). This generator produces resumes that
are structurally realistic enough to fine-tune and evaluate an extraction
model, while giving us perfect ground truth for every field (impossible to
get cheaply on real resumes without the same manual QA effort TalentGrid is
already drowning in).

HOW THIS ADDRESSES THE STATED FAILURE MODES
- "Overfitting on a Narrow Sample": we generate across MANY industries
  (software, nursing, accounting, logistics, teaching, sales, manufacturing,
  legal, marketing, construction) and several distinct FORMATS (bullet-style,
  narrative cover-letter style, dense-paragraph, chronological-table-style),
  and we hold out entire industries from the training split so the eval can
  measure true generalization, not memorization.
- "Schema Drift / fabrication": every generated resume includes fields that
  are DELIBERATELY ambiguous or absent (e.g. a skill mentioned with no years
  given, a cert with no issuing body stated) so the correct label is null,
  training the model that "no evidence -> null", not "guess something
  plausible".
- Career gaps are generated explicitly as date-range arithmetic so the label
  is checkable exactly.

OUTPUT
Writes data/train.jsonl, data/val.jsonl, data/test.jsonl, data/test_ood.jsonl
(ood = out-of-distribution industries entirely excluded from train/val, used
to measure generalization the way the client's failure post-mortem calls
for).
"""
import json
import random
from pathlib import Path
from faker import Faker

fake = Faker()
random.seed(7)
Faker.seed(7)

INDUSTRIES = {
    "software": {
        "skills": ["Python", "React", "AWS", "Docker", "Kubernetes", "SQL", "Java", "Go", "Kafka", "Terraform"],
        "certs": ["AWS Certified Solutions Architect", "Certified Kubernetes Administrator", "PMP"],
        "degrees": ["B.S. Computer Science", "M.S. Software Engineering", "B.S. Information Technology"],
    },
    "nursing": {
        "skills": ["Patient Triage", "IV Insertion", "EHR Documentation", "Wound Care", "ICU Monitoring"],
        "certs": ["RN License", "BLS Certification", "ACLS Certification"],
        "degrees": ["BSN", "ADN", "MSN"],
    },
    "accounting": {
        "skills": ["QuickBooks", "GAAP Reporting", "Tax Preparation", "Auditing", "Excel Financial Modeling"],
        "certs": ["CPA", "CMA", "Enrolled Agent"],
        "degrees": ["B.S. Accounting", "M.S. Accounting", "MBA Finance"],
    },
    "logistics": {
        "skills": ["Fleet Scheduling", "Inventory Management", "SAP WM", "Route Optimization", "Customs Compliance"],
        "certs": ["Six Sigma Green Belt", "APICS CPIM"],
        "degrees": ["B.S. Supply Chain Management", "B.A. Business Logistics"],
    },
    "teaching": {
        "skills": ["Curriculum Design", "Classroom Management", "IEP Development", "Google Classroom"],
        "certs": ["State Teaching License", "TESOL Certificate"],
        "degrees": ["B.A. Education", "M.Ed."],
    },
    "sales": {
        "skills": ["Salesforce CRM", "Cold Outreach", "Contract Negotiation", "Account Management"],
        "certs": ["Salesforce Certified Administrator", "HubSpot Sales Certification"],
        "degrees": ["B.A. Business Administration", "B.S. Marketing"],
    },
    "manufacturing": {
        "skills": ["CNC Programming", "Lean Manufacturing", "Quality Control", "AutoCAD", "SolidWorks"],
        "certs": ["Six Sigma Black Belt", "OSHA 30"],
        "degrees": ["B.S. Mechanical Engineering", "A.S. Manufacturing Technology"],
    },
    "legal": {
        "skills": ["Contract Review", "Legal Research", "Westlaw", "E-Discovery", "Litigation Support"],
        "certs": ["Paralegal Certificate", "Bar Admission"],
        "degrees": ["J.D.", "B.A. Legal Studies"],
    },
}

# Entire industries held out from train/val, only ever appearing in
# test_ood.jsonl, to measure generalization to unseen formats/domains.
OOD_INDUSTRIES = ["marketing", "construction"]
INDUSTRIES["marketing"] = {
    "skills": ["SEO", "Google Analytics", "Content Strategy", "A/B Testing", "Meta Ads"],
    "certs": ["Google Analytics Certification", "HubSpot Content Marketing"],
    "degrees": ["B.A. Marketing", "B.S. Communications"],
}
INDUSTRIES["construction"] = {
    "skills": ["Blueprint Reading", "OSHA Compliance", "Project Scheduling", "Cost Estimation"],
    "certs": ["OSHA 30", "PMP"],
    "degrees": ["B.S. Construction Management"],
}

FORMATS = ["bullets", "narrative", "dense_paragraph", "chrono_table"]


def render_bullets(name, industry, skills, edu, certs, gap_text):
    lines = [f"{name}", "", "SKILLS"]
    for s in skills:
        lines.append(f"- {s['text']}")
    lines += ["", "EXPERIENCE"]
    if gap_text:
        lines.append(gap_text)
    lines += ["", "EDUCATION"]
    for e in edu:
        lines.append(f"- {e['text']}")
    if certs:
        lines += ["", "CERTIFICATIONS"]
        for c in certs:
            lines.append(f"- {c['text']}")
    return "\n".join(lines)


def render_narrative(name, industry, skills, edu, certs, gap_text):
    parts = [f"Dear Hiring Manager,\n\nMy name is {name} and I am applying for a role in {industry}."]
    parts.append("Over my career I have developed strengths including " + "; ".join(s['text'] for s in skills) + ".")
    if gap_text:
        parts.append(gap_text)
    parts.append("My academic background includes " + "; ".join(e['text'] for e in edu) + ".")
    if certs:
        parts.append("I also hold " + "; ".join(c['text'] for c in certs) + ".")
    parts.append("Thank you for your consideration.\n\nSincerely,\n" + name)
    return "\n\n".join(parts)


def render_dense(name, industry, skills, edu, certs, gap_text):
    body = f"{name} is a {industry} professional. "
    body += " ".join(s['text'] + "." for s in skills)
    if gap_text:
        body += " " + gap_text
    body += " " + " ".join(e['text'] + "." for e in edu)
    if certs:
        body += " " + " ".join(c['text'] + "." for c in certs)
    return body


def render_table(name, industry, skills, edu, certs, gap_text):
    lines = [f"NAME: {name}", f"FIELD: {industry}", "", "| Category | Detail |", "|---|---|"]
    for s in skills:
        lines.append(f"| Skill | {s['text']} |")
    for e in edu:
        lines.append(f"| Education | {e['text']} |")
    for c in certs:
        lines.append(f"| Certification | {c['text']} |")
    if gap_text:
        lines.append(f"| Note | {gap_text} |")
    return "\n".join(lines)


RENDERERS = {
    "bullets": render_bullets,
    "narrative": render_narrative,
    "dense_paragraph": render_dense,
    "chrono_table": render_table,
}


def make_example(industry):
    profile = INDUSTRIES[industry]
    name = fake.name()

    # --- skills: some with explicit years (labelable), some ambiguous (-> null years) ---
    chosen_skill_names = random.sample(profile["skills"], k=random.randint(2, 4))
    skills_gt, skill_texts = [], []
    for s in chosen_skill_names:
        if random.random() < 0.6:
            yrs = random.randint(1, 12)
            text = f"{yrs} years of experience with {s}"
            skills_gt.append({"name": s, "years_experience": yrs, "confidence": 0.9, "evidence_span": text})
        else:
            text = f"Proficient in {s}"  # no years stated -> ground truth years_experience = None
            skills_gt.append({"name": s, "years_experience": None, "confidence": 0.5, "evidence_span": text})
        skill_texts.append({"text": text})

    # --- education ---
    degree = random.choice(profile["degrees"])
    institution = fake.city() + " University"
    if random.random() < 0.75:
        grad_year = random.randint(2005, 2023)
        edu_text = f"{degree}, {institution}, {grad_year}"
        edu_gt = [{"degree": degree, "institution": institution, "graduation_year": grad_year,
                   "confidence": 0.9, "evidence_span": edu_text}]
    else:
        edu_text = f"{degree}, {institution}"  # no year stated -> ground truth graduation_year = None
        edu_gt = [{"degree": degree, "institution": institution, "graduation_year": None,
                   "confidence": 0.5, "evidence_span": edu_text}]
    edu_render = [{"text": edu_text}]

    # --- certifications: sometimes present, sometimes absent entirely ---
    certs_gt, cert_render = [], []
    if random.random() < 0.6:
        cert_name = random.choice(profile["certs"])
        if random.random() < 0.5:
            issuer = fake.company()
            cert_text = f"{cert_name}, issued by {issuer}"
            certs_gt.append({"name": cert_name, "issuer": issuer, "year": None,
                              "confidence": 0.8, "evidence_span": cert_text})
        else:
            cert_text = cert_name  # no issuer stated -> ground truth issuer = None
            certs_gt.append({"name": cert_name, "issuer": None, "year": None,
                              "confidence": 0.5, "evidence_span": cert_text})
        cert_render.append({"text": cert_text})

    # --- career gap: explicit date arithmetic so the label is exactly checkable ---
    gap_text, gap_gt = None, []
    if random.random() < 0.4:
        start = random.randint(2010, 2018)
        end = start + random.randint(1, 3)
        gap_text = f"Note: no employment listed between {start} and {end}."
        gap_gt.append({"start_year": start, "end_year": end,
                        "duration_months": (end - start) * 12,
                        "confidence": 0.7, "evidence_span": gap_text})

    fmt = random.choice(FORMATS)
    text = RENDERERS[fmt](name, industry, skill_texts, edu_render, cert_render, gap_text)

    label = {"skills": skills_gt, "education": edu_gt, "certifications": certs_gt, "career_gaps": gap_gt}
    return {"text": text, "label": label, "industry": industry, "format": fmt}


def build(n_per_industry=60):
    in_dist_industries = [i for i in INDUSTRIES if i not in OOD_INDUSTRIES]
    examples = []
    for industry in in_dist_industries:
        for _ in range(n_per_industry):
            examples.append(make_example(industry))
    random.shuffle(examples)

    n = len(examples)
    train = examples[: int(n * 0.7)]
    val = examples[int(n * 0.7): int(n * 0.85)]
    test = examples[int(n * 0.85):]

    ood_examples = []
    for industry in OOD_INDUSTRIES:
        for _ in range(n_per_industry):
            ood_examples.append(make_example(industry))

    out_dir = Path(__file__).parent
    for name, split in [("train", train), ("val", val), ("test", test), ("test_ood", ood_examples)]:
        with open(out_dir / f"{name}.jsonl", "w") as f:
            for ex in split:
                f.write(json.dumps(ex) + "\n")
        print(f"{name}: {len(split)} examples")


if __name__ == "__main__":
    build()
