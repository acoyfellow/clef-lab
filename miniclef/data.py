"""Synthetic Clef-style records: a latent world -> state text -> typed questions with gold answers.
Mirrors Clef's described recipe (synthetic data, permuted field order, prompts and schema structure).
Some question families are held out entirely so eval measures schema generalization."""
import random

PRODUCTS = {
    "billing": ["my invoice", "the charge on my card", "our subscription renewal", "a duplicate payment"],
    "dns": ["our DNS records", "the nameserver change", "domain resolution", "the CNAME we added"],
    "workers": ["my Worker script", "the deploy of our Worker", "a Worker throwing exceptions", "wrangler deploy"],
    "storage": ["our R2 bucket", "object uploads", "the storage bucket", "files we stored"],
    "access": ["the login page", "Zero Trust access policies", "SSO sign-in", "our Access app"],
}
PRODUCT_DESC = {
    "billing": ["payments, invoices and refunds", "money, charges and subscriptions", "anything about being billed"],
    "dns": ["domain names and DNS records", "name resolution and nameservers", "DNS configuration"],
    "workers": ["serverless code and deploys", "Workers scripts and runtime errors", "code running on the edge"],
    "storage": ["object storage and buckets", "stored files and uploads", "R2 storage"],
    "access": ["logins, SSO and access policies", "authentication and Zero Trust", "who can sign in"],
}
URGENCY_TEXT = [
    ["whenever you get a chance", "no rush on this", "just curious"],
    ["it would be nice to fix this week", "this is a bit annoying", "please look when you can"],
    ["this is blocking part of our team", "we need this fixed today", "this is hurting us"],
    ["everything is down for all customers", "production is completely broken", "we are losing money every minute"],
]
TONE = {
    "angry": ["This is unacceptable.", "I am extremely frustrated.", "Honestly this is ridiculous."],
    "neutral": ["Here are the details.", "Thanks.", "Let me know what you need."],
    "happy": ["Love the product otherwise!", "You folks are great.", "Thanks so much, really appreciate it!"],
}
TIERS = ["free", "pro", "business", "enterprise"]
TIER_TEXT = {"free": ["on the free plan", "a free account"], "pro": ["on Pro", "a Pro subscriber"],
             "business": ["on the Business plan", "a Business customer"], "enterprise": ["an Enterprise customer", "on our Enterprise contract"]}
FILLER = ["Our site gets a lot of traffic.", "We migrated last month.", "I checked the docs already.",
          "My colleague also saw this.", "We use Terraform for config.", "The dashboard looks normal otherwise."]


def world(rng):
    w = {"product": rng.choice(list(PRODUCTS)), "urgency": rng.randrange(4), "tone": rng.choice(list(TONE)),
         "tier": rng.choice(TIERS), "refund": rng.random() < 0.35,
         "amount": rng.choice([None, None, rng.choice([9, 20, 49, 120, 300, 850, 2400])]),
         "error_code": rng.choice([None, None, rng.choice([500, 502, 403, 1101, 1015])])}
    w["outage"] = w["urgency"] == 3
    return w


def state_text(w, rng):
    parts = [f"Hi, I'm {rng.choice(TIER_TEXT[w['tier']])} and I have a problem with {rng.choice(PRODUCTS[w['product']])}.",
             rng.choice(URGENCY_TEXT[w["urgency"]]).capitalize() + "."]
    if w["amount"] is not None:
        parts.append(rng.choice([f"We were charged ${w['amount']}.", f"The amount involved is ${w['amount']}.", f"It cost us ${w['amount']} so far."]))
    if w["error_code"] is not None:
        parts.append(rng.choice([f"We keep seeing error {w['error_code']}.", f"The response is a {w['error_code']}."]))
    if w["refund"]:
        parts.append(rng.choice(["I would like a refund.", "Please refund us.", "Can we get our money back?"]))
    parts += rng.sample(FILLER, rng.randrange(0, 3))
    parts.append(rng.choice(TONE[w["tone"]]))
    head, rest = parts[:1], parts[1:]
    rng.shuffle(rest)
    return " ".join(head + rest)


def rand_id(rng):
    return rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") + str(rng.randrange(10, 99))


def q_team(w, rng, obscure):
    keys = list(PRODUCTS); rng.shuffle(keys)
    ids = {k: (rand_id(rng) if obscure else k) for k in keys}
    return {"type": "choice", "instructions": rng.choice(["Which team should handle this?", "Route this ticket to a team.", "Which area is this about?"]),
            "criteria": {ids[k]: rng.choice(PRODUCT_DESC[k]) for k in keys}}, ids[w["product"]]


def q_urgency(w, rng, obscure):
    levels = [["Can wait", "Low"], ["This week", "Medium"], ["Today", "High"], ["Right now, outage", "Critical"]]
    return {"type": "score", "instructions": rng.choice(["How urgent is this?", "How quickly must we respond?", "Rate the urgency."]),
            "criteria": [rng.choice(l) for l in levels]}, str(w["urgency"])


def q_outage(w, rng, obscure):
    return {"type": "noul", "instructions": rng.choice(["Is a service fully down?", "Is this an outage?", "Is production broken for everyone?"])}, ("true" if w["outage"] else "false")


def q_refund(w, rng, obscure):
    return {"type": "noul", "instructions": rng.choice(["Does the customer ask for a refund?", "Is money being requested back?", "Refund requested?"])}, ("true" if w["refund"] else "false")


def q_tier(w, rng, obscure):
    ts = TIERS[:]; rng.shuffle(ts)
    ids = {t: (rand_id(rng) if obscure else t) for t in ts}
    return {"type": "choice", "instructions": rng.choice(["Which plan is the customer on?", "What is the customer's tier?"]),
            "criteria": {ids[t]: f"the {t} plan" for t in ts}}, ids[w["tier"]]


# Held out from training: never seen as a family.
def q_tone(w, rng, obscure):
    ts = list(TONE); rng.shuffle(ts)
    desc = {"angry": "upset, frustrated or angry", "neutral": "matter-of-fact, no strong feeling", "happy": "positive, friendly or grateful"}
    ids = {t: (rand_id(rng) if obscure else t) for t in ts}
    return {"type": "choice", "instructions": "What is the customer's tone?", "criteria": {ids[t]: desc[t] for t in ts}}, ids[w["tone"]]


def q_big_amount(w, rng, obscure):
    gold = w["amount"] is not None and w["amount"] > 100
    return {"type": "noul", "instructions": "Does the message mention an amount over 100 dollars?"}, ("true" if gold else "false")


def q_error(w, rng, obscure):
    return {"type": "noul", "instructions": "Does the message include an HTTP or error code?"}, ("true" if w["error_code"] is not None else "false")


TRAIN_FAMILIES = {"team": q_team, "urgency": q_urgency, "outage": q_outage, "refund": q_refund, "tier": q_tier}
HELDOUT_FAMILIES = {"tone": q_tone, "big_amount": q_big_amount, "error_code": q_error}


def make_record(rng, families, obscure_p=0.3, k=(2, 4)):
    w = world(rng)
    names = rng.sample(list(families), min(len(families), rng.randint(*k)))
    questions, gold = {}, {}
    for name in names:
        q, g = families[name](w, rng, rng.random() < obscure_p)
        qid = rng.choice([name, f"f_{rand_id(rng).lower()}", f"{name}_{rng.randrange(9)}"])
        questions[qid], gold[qid] = q, g
    return {"state": state_text(w, rng), "questions": questions, "gold": gold, "families": dict(zip(questions, names))}


def dataset(n, seed, families, **kw):
    rng = random.Random(seed)
    return [make_record(rng, families, **kw) for _ in range(n)]
