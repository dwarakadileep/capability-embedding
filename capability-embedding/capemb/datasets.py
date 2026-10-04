"""Formally specified application problems (Deliverable 3).  Running this module writes data/*.json."""
from __future__ import annotations

import json
import os

from .model import Capability, Goal, State

UUID = "valid UUIDs"


def cap(name, type, pre=(), eff=(), inputs=(), outputs=(), constraints=(), resources=(), cost=None,
        rel=1.0, avail=1.0, window=None, mech=None, group=None, role="relevant"):
    return dict(name=name, type=type, pre=list(pre), eff=list(eff), inputs=[list(i) for i in inputs],
                outputs=[list(o) for o in outputs], constraints=list(constraints), resources=list(resources),
                cost=cost or {}, reliability=rel, availability=avail, window=window, mechanism=mech or {},
                group=group, role=role)


WARM = {"User.authenticated": True, "User.role": "CUSTOMER", "Cart.exists": True, "Cart.item_count": 3,
        "Order.exists": False, "Payment.status": "NOT_STARTED", "Inventory.available": True,
        "Notification.sent": False, "Payment.amount_within_limit": True, "Quantity.valid": True}
COLD = {"User.authenticated": False, "User.role": "CUSTOMER", "Cart.exists": False, "Cart.item_count": 0,
        "Order.exists": False, "Payment.status": "NOT_STARTED", "Inventory.available": True,
        "Notification.sent": False, "Payment.amount_within_limit": True, "Quantity.valid": True}

ECOMMERCE = dict(
    name="ecommerce",
    goal=["Order.exists==true", "Payment.status==SUCCESS", "Notification.sent==true"],
    states={"warm": dict(vars=WARM, data=["cart_id:UUID", "auth_token:TOKEN", "item_id:UUID", "payment_method:STRING"]),
            "cold": dict(vars=COLD, data=["credentials:STRING", "item_id:UUID", "payment_method:STRING"])},
    capabilities=[
        cap("Authenticate", "API", eff=["User.authenticated==true"], inputs=[("credentials", "STRING", "any", True)],
            outputs=[("auth_token", "TOKEN", "jwt")], resources=["Auth service", "Network"],
            cost=dict(time_ms=150, money=0.0, resource=1, risk=0.02), rel=0.995,
            mech=dict(method="POST", endpoint="/login")),
        cap("CreateCart", "API", pre=["User.authenticated==true"], eff=["Cart.exists==true"],
            inputs=[("auth_token", "TOKEN", "jwt", True)], outputs=[("cart_id", "UUID", UUID)],
            resources=["Database", "Network"], cost=dict(time_ms=80, resource=1, risk=0.01), rel=0.998,
            mech=dict(method="POST", endpoint="/carts")),
        cap("AddItem", "API", pre=["Cart.exists==true"], eff=["Cart.item_count==1"],
            inputs=[("cart_id", "UUID", UUID, True), ("item_id", "UUID", UUID, True)],
            constraints=["Quantity.valid==true"], resources=["Database", "Network"],
            cost=dict(time_ms=90, resource=1, risk=0.01), rel=0.997, mech=dict(method="POST", endpoint="/carts/items")),
        cap("CreateOrder_API", "API", group="create_order",
            pre=["User.authenticated==true", "Cart.exists==true", "Cart.item_count>0", "Inventory.available==true"],
            eff=["Order.exists==true", "Order.status==CREATED", "Cart.locked==true"],
            inputs=[("cart_id", "UUID", UUID, True)], outputs=[("order_id", "UUID", UUID)],
            constraints=["User.role in {CUSTOMER,ADMIN}"], resources=["Database", "Network"],
            cost=dict(time_ms=120, money=0.001, resource=2, risk=0.02), rel=0.99,
            mech=dict(method="POST", endpoint="/orders")),
        cap("CreateOrder_DB", "DATABASE", group="create_order",
            pre=["User.authenticated==true", "Cart.exists==true", "Cart.item_count>0", "Inventory.available==true"],
            eff=["Order.exists==true", "Order.status==CREATED", "Cart.locked==true"],
            inputs=[("cart_id", "UUID", UUID, True)], outputs=[("order_id", "UUID", UUID)],
            constraints=["User.role in {CUSTOMER,ADMIN}"], resources=["Database"],
            cost=dict(time_ms=15, money=0.0, resource=1, risk=0.05), rel=0.97,
            mech=dict(operation="INSERT", table="orders")),
        cap("CreateOrder_GUI", "GUI", group="create_order",
            pre=["User.authenticated==true", "Cart.exists==true", "Cart.item_count>0", "Inventory.available==true"],
            eff=["Order.exists==true", "Order.status==CREATED", "Cart.locked==true"],
            inputs=[("cart_id", "UUID", UUID, True)], outputs=[("order_id", "UUID", UUID)],
            constraints=["User.role in {CUSTOMER,ADMIN}"], resources=["Network"],
            cost=dict(time_ms=2500, money=0.0, resource=3, risk=0.15), rel=0.90,
            mech=dict(action="CLICK", component="submit_button")),
        cap("MakePayment_Gateway", "API", group="payment", pre=["Order.exists==true"], eff=["Payment.status==SUCCESS"],
            inputs=[("order_id", "UUID", UUID, True), ("payment_method", "STRING", "any", True)],
            outputs=[("payment_id", "UUID", UUID)], constraints=["Payment.amount_within_limit==true"],
            resources=["Payment gateway", "Network"], cost=dict(time_ms=800, money=0.03, resource=2, risk=0.05),
            rel=0.95, mech=dict(method="POST", endpoint="/payments")),
        cap("MakePayment_Backup", "API", group="payment", pre=["Order.exists==true"], eff=["Payment.status==SUCCESS"],
            inputs=[("order_id", "UUID", UUID, True), ("payment_method", "STRING", "any", True)],
            outputs=[("payment_id", "UUID", UUID)], constraints=["Payment.amount_within_limit==true"],
            resources=["Payment gateway", "Network"], cost=dict(time_ms=1500, money=0.10, resource=2, risk=0.02),
            rel=0.995, mech=dict(method="POST", endpoint="/payments/backup")),
        cap("MakePayment_Bank", "SERVICE", group="payment", pre=["Order.exists==true"], eff=["Payment.status==SUCCESS"],
            inputs=[("order_id", "UUID", UUID, True), ("payment_method", "STRING", "any", True)],
            outputs=[("payment_id", "UUID", UUID)], constraints=["Payment.amount_within_limit==true"],
            resources=["External service", "Network"], cost=dict(time_ms=600, money=0.01, resource=2, risk=0.08),
            rel=0.999, window=[9, 17], mech=dict(service="bank_transfer", mode="batch")),
        cap("SendNotification", "EVENT", pre=["Payment.status==SUCCESS"], eff=["Notification.sent==true"],
            inputs=[("order_id", "UUID", UUID, True)], resources=["Network"],
            cost=dict(time_ms=60, money=0.0005, resource=1, risk=0.01), rel=0.99,
            mech=dict(trigger="PaymentSucceeded", handler="SendNotification")),
        cap("CancelCart", "API", pre=["Order.exists==false", "Cart.exists==true"], eff=["Cart.exists==false"],
            inputs=[("cart_id", "UUID", UUID, True)], resources=["Database"], role="irrelevant",
            cost=dict(time_ms=50, resource=1, risk=0.01), rel=0.998, mech=dict(method="DELETE", endpoint="/carts")),
        cap("GenerateReport", "COMPUTATION", pre=["Order.exists==true"], eff=["Report.generated==true"],
            inputs=[("order_id", "UUID", UUID, True)], outputs=[("report_id", "UUID", UUID)], role="irrelevant",
            resources=["Database"], cost=dict(time_ms=900, resource=3, energy=2, risk=0.01), rel=0.99,
            mech=dict(function="build_report")),
        cap("UpdateProfile", "API", pre=["User.authenticated==true"], eff=["User.profile_updated==true"], role="irrelevant",
            resources=["Database", "Network"], cost=dict(time_ms=70, resource=1, risk=0.01), rel=0.998,
            mech=dict(method="PUT", endpoint="/profile")),
        cap("SendMarketingEmail", "MESSAGE", pre=["User.authenticated==true"], eff=["Marketing.sent==true"], role="irrelevant",
            resources=["Network"], cost=dict(time_ms=40, money=0.002, resource=1, risk=0.03), rel=0.98,
            mech=dict(channel="email", template="promo")),
        cap("ExportLogs", "FILE", eff=["Logs.exported==true"], outputs=[("log_file", "PATH", "any")], role="irrelevant",
            resources=["File system"], cost=dict(time_ms=300, resource=1, risk=0.01), rel=0.999,
            mech=dict(path="/var/log", format="csv")),
    ],
    chains=[["CreateOrder_API", "MakePayment_Gateway", "SendNotification"],
            ["CreateOrder_DB", "MakePayment_Backup", "SendNotification"],
            ["Authenticate", "CreateCart", "AddItem", "CreateOrder_API", "MakePayment_Gateway", "SendNotification"]],
)

FILES = dict(
    name="filepipeline",
    goal=["File.archived==true", "Owner.notified==true"],
    states={"start": dict(vars={"File.uploaded": False, "Storage.free": True, "Owner.known": True}, data=["raw_bytes:BYTES"])},
    capabilities=[
        cap("UploadFile", "API", pre=["Storage.free==true"], eff=["File.uploaded==true"],
            inputs=[("raw_bytes", "BYTES", "any", True)], outputs=[("file_id", "UUID", UUID)], resources=["Network"],
            cost=dict(time_ms=400, resource=1, risk=0.03), rel=0.98, mech=dict(method="PUT", endpoint="/files")),
        cap("ValidateFile", "FUNCTION", pre=["File.uploaded==true"], eff=["File.valid==true"],
            inputs=[("file_id", "UUID", UUID, True)], cost=dict(time_ms=40, resource=1), rel=0.999,
            mech=dict(function="validate")),
        cap("ConvertToPDF", "COMPUTATION", pre=["File.valid==true"], eff=["File.format==PDF"],
            inputs=[("file_id", "UUID", UUID, True)], outputs=[("pdf_id", "UUID", UUID)], resources=["CPU"],
            cost=dict(time_ms=1200, resource=4, energy=3, risk=0.04), rel=0.96, mech=dict(function="convert", to="pdf")),
        cap("StoreArchive", "DATABASE", pre=["File.format==PDF"], eff=["File.archived==true"],
            inputs=[("pdf_id", "UUID", UUID, True)], resources=["Database"], cost=dict(time_ms=30, resource=1, risk=0.01),
            rel=0.995, mech=dict(operation="INSERT", table="archive")),
        cap("NotifyOwner", "MESSAGE", pre=["File.archived==true", "Owner.known==true"], eff=["Owner.notified==true"],
            resources=["Network"], cost=dict(time_ms=60, money=0.0005, resource=1), rel=0.99,
            mech=dict(channel="email", template="archived")),
        cap("ResizeImage", "COMPUTATION", pre=["File.valid==true"], eff=["File.format==PNG"], role="irrelevant",
            inputs=[("file_id", "UUID", UUID, True)], outputs=[("png_id", "UUID", UUID)], resources=["CPU"],
            cost=dict(time_ms=500, resource=2, energy=1), rel=0.98, mech=dict(function="resize", to="png")),
        cap("DeleteFile", "FILE", pre=["File.uploaded==true"], eff=["File.uploaded==false"], role="irrelevant",
            inputs=[("file_id", "UUID", UUID, True)], resources=["File system"], cost=dict(time_ms=20, resource=1, risk=0.2),
            rel=0.999, mech=dict(op="unlink")),
        cap("ScanVirus", "SERVICE", pre=["File.uploaded==true"], eff=["File.scanned==true"], role="irrelevant",
            inputs=[("file_id", "UUID", UUID, True)], resources=["External service"], cost=dict(time_ms=2000, money=0.02, resource=2),
            rel=0.97, avail=1.0, mech=dict(service="av", mode="sync")),
    ],
    chains=[["UploadFile", "ValidateFile", "ConvertToPDF", "StoreArchive", "NotifyOwner"]],
)

# Exactly the three capabilities of Required Experiment 1.
PATTERN = dict(
    name="assignment_pattern",
    goal=["Payment.status==SUCCESS"],
    states={"start": dict(vars={"Order.exists": False, "Payment.status": "NOT_STARTED"}, data=[])},
    capabilities=[
        cap("CreateOrder", "API", eff=["Order.exists==true"], mech=dict(method="POST", endpoint="/orders")),
        cap("MakePayment", "API", pre=["Order.exists==true"], eff=["Payment.status==SUCCESS"],
            mech=dict(method="POST", endpoint="/payments")),
        cap("CancelCart", "API", pre=["Order.exists==false"], eff=["Cart.exists==false"],
            mech=dict(method="DELETE", endpoint="/carts")),
    ],
    chains=[["CreateOrder", "MakePayment"]],
)

ALL = {"ecommerce": ECOMMERCE, "filepipeline": FILES, "assignment_pattern": PATTERN}


class Problem:
    def __init__(self, d: dict):
        self.name = d["name"]
        self.caps = {c["name"]: Capability(
            name=c["name"], type=c["type"], inputs=c["inputs"], outputs=c["outputs"], pre=c["pre"], eff=c["eff"],
            constraints=c["constraints"], resources=c["resources"], cost=c["cost"], reliability=c["reliability"],
            availability=c["availability"], window=tuple(c["window"]) if c["window"] else None,
            mechanism=c["mechanism"]) for c in d["capabilities"]}
        self.meta = {c["name"]: dict(group=c.get("group"), role=c.get("role", "relevant")) for c in d["capabilities"]}
        self.goal = Goal(d["goal"])
        self.states = {k: State(dict(v["vars"]), set(v["data"])) for k, v in d["states"].items()}
        self.chains = d["chains"]

    def irrelevant(self):
        return [n for n, m in self.meta.items() if m["role"] == "irrelevant"]


def dump(path="data"):
    os.makedirs(path, exist_ok=True)
    for k, d in ALL.items():
        with open(os.path.join(path, f"{k}.json"), "w") as f:
            json.dump(d, f, indent=1)


def load(name: str, path="data") -> Problem:
    with open(os.path.join(path, f"{name}.json")) as f:
        return Problem(json.load(f))


if __name__ == "__main__":
    dump()
    print("wrote", list(ALL))
