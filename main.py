import base64
import io
import json
import pathlib
import re
import uuid
import secrets
from datetime import datetime, date, time, timezone, timedelta
from typing import Optional, Callable, Dict, Any, Set, List

from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from sqlalchemy import create_engine, Column, String, DateTime, Text, Boolean, Date, func, or_
from sqlalchemy.orm import sessionmaker, declarative_base
from cryptography.fernet import Fernet, InvalidToken
import qrcode
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# ----------------------------
# Configuración / Settings
# ----------------------------
class Settings(BaseSettings):
    SECRET_KEY: Optional[str] = None
    DATABASE_URL: str = "sqlite:///./tickets.db"
    TOKEN_TTL_HOURS: int = 8
    RESET_DB_ON_DEPLOY: bool = False   # 👈 NUEVO

    class Config:
        env_file = ".env"

settings = Settings()

if not settings.SECRET_KEY:
    settings.SECRET_KEY = Fernet.generate_key().decode("utf-8")
fernet = Fernet(settings.SECRET_KEY.encode("utf-8"))

def maybe_reset_sqlite():
    if not settings.RESET_DB_ON_DEPLOY:
        return
    if not settings.DATABASE_URL.startswith("sqlite"):
        return

    m = re.match(r"^sqlite:///(.+)$", settings.DATABASE_URL)
    if not m:
        return
    db_path = pathlib.Path(m.group(1)).resolve()
    if db_path.exists():
        db_path.unlink()
        print(f"[reset] Base de datos SQLite eliminada: {db_path}")

maybe_reset_sqlite()

# ----------------------------
# Usuarios demo + permisos
# ----------------------------
USER_DB: Dict[str, Dict[str, Any]] = {
    "admin_lobo": {
    "password": "L0b0#Adm!n2025",
    "perms": [
            "health:read",
            "tickets:create",
            "tickets:validate",
            "tickets:read_status",
            "tickets:list"
        ]
    },
    "admin_aguila": {
    "password": "AgU1l@Adm#45",
    "perms": [
            "health:read",
            "tickets:create",
            "tickets:validate",
            "tickets:read_status",
             "tickets:list"
        ]
    },
    "taquilla_tigre": {
        "password": "T1gr3#Scan2025",
        "perms": [
            "health:read",
            "tickets:validate",
            "tickets:read_status",
            "tickets:list"
        ]
    },
    "taquilla_oso": {
        "password": "0s0#Taq@78",
        "perms": [
            "health:read",
            "tickets:validate",
            "tickets:read_status",
            "tickets:list"
        ]
    },
    "viewer_delfin": {
        "password": "D3lf!n#View25",
        "perms": [
            "health:read",
            "tickets:read_status",
            "tickets:list"
        ]
    },
    "viewer_zorro": {
        "password": "Z0rr0!Lst#9",
        "perms": [
            "health:read",
            "tickets:read_status",
            "tickets:list"
        ]
    }
}


TOKENS: Dict[str, Dict[str, Any]] = {}
bearer_scheme = HTTPBearer(auto_error=True)

def issue_token(username: str) -> str:
    token = uuid.uuid4().hex + secrets.token_hex(16)
    TOKENS[token] = {
        "user": username,
        "perms": set(USER_DB[username]["perms"]),
        "exp": datetime.now(timezone.utc) + timedelta(hours=settings.TOKEN_TTL_HOURS),
    }
    return token

def get_token_payload(credentials: HTTPAuthorizationCredentials) -> Dict[str, Any]:
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Esquema de autorización inválido")
    token = credentials.credentials
    data = TOKENS.get(token)
    if not data:
        raise HTTPException(status_code=401, detail="Token inválido")
    if data["exp"] < datetime.now(timezone.utc):
        TOKENS.pop(token, None)
        raise HTTPException(status_code=401, detail="Token expirado")
    return data

def require_perm(required_perm: str) -> Callable:
    def _dep(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> Dict[str, Any]:
        payload = get_token_payload(credentials)
        perms: Set[str] = payload.get("perms", set())
        if required_perm not in perms:
            raise HTTPException(status_code=403, detail=f"Permiso faltante: {required_perm}")
        return payload
    return _dep


# ----------------------------
# DB (SQLAlchemy)
# ----------------------------
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Ticket(Base):
    __tablename__ = "tickets"
    id = Column(String, primary_key=True)                  # UUID
    purchaser_name = Column(String, nullable=False)
    event_id = Column(String, nullable=False)
    national_id = Column(String, nullable=False)
    phone = Column(String, nullable=True)

    # NUEVO: fecha + hora del evento (UTC aware)
    event_at = Column(DateTime(timezone=True), nullable=True)

    # Conservado por compatibilidad y filtros rápidos
    event_date = Column(Date, nullable=False)

    created_at = Column(DateTime(timezone=True), nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    used_at = Column(DateTime(timezone=True), nullable=True)
    is_used = Column(Boolean, default=False, nullable=False)
    qr_ciphertext = Column(Text, nullable=False)
    version = Column(String, default="v1", nullable=False)

    created_by = Column(String, nullable=False)            # usuario que creó
    validated_by = Column(String, nullable=True)           # usuario que validó (si aplica)

Base.metadata.create_all(bind=engine)

# ----------------------------
# Modelos de entrada/salida
# ----------------------------
class CreateTicketRequest(BaseModel):
    purchaser_name: str = Field(..., description="Nombre del invitado")
    event_id: str = Field(..., description="ID de evento")
    national_id: str = Field(..., description="Cédula")
    phone: Optional[str] = Field(None, description="Teléfono (opcional)")
    event_date: date = Field(..., description="Fecha del evento (YYYY-MM-DD)")
    # NUEVO: permite especificar la hora exacta del evento
    event_at: Optional[datetime] = Field(None, description="Fecha y hora del evento (ISO8601, recomendado)")
    expires_at: Optional[datetime] = Field(None, description="Vencimiento del QR (UTC, opcional)")

class CreateTicketResponse(BaseModel):
    ticket_id: str
    qr_ciphertext: str
    qr_png_base64: str
    expires_at: Optional[datetime]
    purchaser_name: str
    event_id: str
    national_id: str
    phone: Optional[str]
    event_date: date
    event_at: datetime

class ValidateTicketRequest(BaseModel):
    qr: str = Field(..., description="Contenido tal cual del QR (ciphertext)")

class ValidateTicketResponse(BaseModel):
    valid: bool
    reason: Optional[str] = None
    ticket_id: Optional[str] = None
    purchaser_name: Optional[str] = None
    event_id: Optional[str] = None
    event_date: Optional[date] = None
    national_id: Optional[str] = None
    phone: Optional[str] = None
    used_at: Optional[datetime] = None

class TicketStatusResponse(BaseModel):
    ticket_id: str
    valid: bool
    is_used: bool
    purchaser_name: str
    event_id: str
    national_id: str
    phone: Optional[str]
    event_date: date
    event_at: datetime
    created_at: datetime
    expires_at: Optional[datetime]
    used_at: Optional[datetime]

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_at: datetime

class TicketListItem(BaseModel):
    n: int                        # enumeración (1..)
    ticket_id: str
    purchaser_name: str
    event_id: str
    national_id: str
    phone: Optional[str]
    event_date: date
    event_at: datetime
    created_at: datetime
    expires_at: Optional[datetime]
    used_at: Optional[datetime]
    is_used: bool
    created_by: str
    validated_by: Optional[str]

class TicketListResponse(BaseModel):
    items: List[TicketListItem]
    total: int
    used: int
    unused: int
    page: int
    page_size: int

class TicketSummaryResponse(BaseModel):
    total: int
    used: int
    unused: int

# ----------------------------
# Utilidades de cifrado / QR
# ----------------------------
def encrypt_payload(payload: dict) -> str:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    token: bytes = fernet.encrypt(data)
    return token.decode("utf-8")

def decrypt_payload(ciphertext: str) -> dict:
    data = fernet.decrypt(ciphertext.encode("utf-8"))
    return json.loads(data.decode("utf-8"))

def make_qr_png_base64(text: str) -> str:
    img = qrcode.make(text)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def _utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)

# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(
    title="QR Tickets API (cifrado)",
    version="1.3.0",
    description="""
API para crear tickets cifrados en QR y validarlos (uso único).
Incluye auth por token Bearer y permisos por endpoint.
Soporta fecha y hora del evento (event_at).
""",
    docs_url = "/swagger",
    swagger_ui_parameters = {"defaultModelsExpandDepth": -1},  # Oculta schemas al abrir
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod: restringe a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Auth
# ----------------------------
@app.post("/login", response_model=LoginResponse, tags=["auth"])
def login(body: LoginRequest):
    user = USER_DB.get(body.username)
    if not user or not secrets.compare_digest(user["password"], body.password):
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    token = issue_token(body.username)
    exp = TOKENS[token]["exp"]
    return LoginResponse(access_token=token, expires_at=exp)

@app.post("/logout", tags=["auth"])
def logout(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = credentials.credentials
    TOKENS.pop(token, None)
    return {"ok": True}

@app.get("/me", tags=["auth"])
def whoami(payload: Dict[str, Any] = Depends(require_perm("health:read"))):
    return {"user": payload["user"], "perms": sorted(list(payload["perms"]))}

# ----------------------------
# Util
# ----------------------------
@app.get("/health", tags=["util"])
def health():
    return {"status": "ok", "algo": "qr-tickets", "version": "1.3.0"}

# ----------------------------
# Tickets
# ----------------------------
@app.post("/tickets", response_model=CreateTicketResponse, tags=["tickets"])
def create_ticket(req: CreateTicketRequest, _ : Dict[str, Any] = Depends(require_perm("tickets:create"))):
    now = datetime.now(timezone.utc)
    ticket_id = str(uuid.uuid4())
    created_by = _["user"]

    # Normaliza fechas
    exp_dt = _to_utc(req.expires_at)

    if req.event_at is not None:
        event_at = _to_utc(req.event_at)
    else:
        # Hora por defecto: 20:00 UTC sobre event_date
        default_hour = time(20, 0, 0)
        event_at = datetime.combine(req.event_date, default_hour).replace(tzinfo=timezone.utc)

    # Payload cifrado que va dentro del QR
    payload = {
        "version": "v1",
        "ticket_id": ticket_id,
        "purchaser_name": req.purchaser_name,
        "event_id": req.event_id,
        "national_id": req.national_id,
        "phone": req.phone,                               # puede ser None
        "event_date": req.event_date.isoformat(),         # "YYYY-MM-DD"
        "event_at": event_at.isoformat().replace("+00:00", "Z"),
        "iat": int(now.timestamp()),
        "exp": int(exp_dt.timestamp()) if exp_dt else None,
    }

    ciphertext = encrypt_payload(payload)
    qr_png = make_qr_png_base64(ciphertext)

    db = SessionLocal()
    try:
        t = Ticket(
            id=ticket_id,
            purchaser_name=req.purchaser_name,
            event_id=req.event_id,
            national_id=req.national_id,
            phone=req.phone,
            event_date=req.event_date,
            event_at=event_at,
            created_at=now,
            expires_at=exp_dt,
            used_at=None,
            is_used=False,
            qr_ciphertext=ciphertext,
            version="v1",
            created_by=created_by,
            validated_by=None,
        )
        db.add(t)
        db.commit()
    finally:
        db.close()

    return CreateTicketResponse(
        ticket_id=ticket_id,
        qr_ciphertext=ciphertext,
        qr_png_base64=qr_png,
        expires_at=exp_dt,
        purchaser_name=req.purchaser_name,
        event_id=req.event_id,
        national_id=req.national_id,
        phone=req.phone,
        event_date=req.event_date,
        event_at=event_at,
    )

@app.post("/validate", response_model=ValidateTicketResponse, tags=["tickets"])
def validate_ticket(req: ValidateTicketRequest, _ : Dict[str, Any] = Depends(require_perm("tickets:validate"))):
    # 1) Descifrar
    validator = _["user"]
    try:
        payload = decrypt_payload(req.qr)
    except InvalidToken:
        return ValidateTicketResponse(valid=False, reason="QR inválido o manipulado")

    # 2) Campos mínimos
    ticket_id = payload.get("ticket_id")
    version = payload.get("version")
    if version != "v1" or not ticket_id:
        return ValidateTicketResponse(valid=False, reason="Formato de payload inválido")

    # 3) Buscar en DB
    db = SessionLocal()
    try:
        t: Ticket = db.get(Ticket, ticket_id)
        if not t:
            return ValidateTicketResponse(valid=False, reason="Ticket no existe")

        # 4) Coincidencia exacta del ciphertext
        if t.qr_ciphertext != req.qr:
            return ValidateTicketResponse(valid=False, reason="QR no coincide con registro")

        # 5) Expiración (normalizada)
        now = datetime.now(timezone.utc)
        exp_dt = _utc(t.expires_at)
        if exp_dt and now > exp_dt:
            return ValidateTicketResponse(valid=False, reason="Ticket expirado", ticket_id=t.id)

        # 6) Ya usado
        if t.is_used:
            return ValidateTicketResponse(
                valid=False,
                reason="Ticket ya fue usado",
                ticket_id=t.id,
                purchaser_name=t.purchaser_name,
                event_id=t.event_id,
                event_date=t.event_date,
                national_id=t.national_id,
                phone=t.phone,
                used_at=_utc(t.used_at),
            )

        # 7) Marcar como usado (primera validación)
        t.is_used = True
        t.used_at = now
        t.validated_by = validator
        db.add(t)
        db.commit()

        return ValidateTicketResponse(
            valid=True,
            ticket_id=t.id,
            purchaser_name=t.purchaser_name,
            event_id=t.event_id,
            event_date=t.event_date,
            national_id=t.national_id,
            phone=t.phone,
            used_at=now,
        )
    finally:
        db.close()

# Rutas estáticas antes de la dinámica {ticket_id}
@app.get("/tickets/summary", response_model=TicketSummaryResponse, tags=["tickets"])
def tickets_summary(_=Depends(require_perm("tickets:list"))):
    db = SessionLocal()
    try:
        total = db.query(func.count(Ticket.id)).order_by(None).scalar() or 0
        used = db.query(func.count(Ticket.id)).filter(Ticket.is_used.is_(True)).order_by(None).scalar() or 0
        unused = total - used
        return TicketSummaryResponse(total=total, used=used, unused=unused)
    finally:
        db.close()

@app.get("/tickets/list", response_model=TicketListResponse, tags=["tickets"])
def tickets_list(
    page: int = 1,
    page_size: int = 50,
    used: Optional[bool] = None,
    q: Optional[str] = None,
    _=Depends(require_perm("tickets:list")),
):
    page = max(1, page)
    page_size = min(500, max(1, page_size))
    offset = (page - 1) * page_size

    db = SessionLocal()
    try:
        base = db.query(Ticket)

        if used is not None:
            base = base.filter(Ticket.is_used.is_(used))

        if q:
            like = f"%{q}%"
            base = base.filter(or_(
                Ticket.purchaser_name.ilike(like),
                Ticket.national_id.ilike(like),
                Ticket.event_id.ilike(like),
                Ticket.id.ilike(like),
            ))

        # Conteos filtrados
        total = base.order_by(None).count()
        used_count = base.filter(Ticket.is_used.is_(True)).order_by(None).count()
        unused_count = base.filter(Ticket.is_used.is_(False)).order_by(None).count()

        # Ordena por event_at (hora del evento), luego created_at
        rows = (base
                .order_by(Ticket.event_at.desc(), Ticket.created_at.desc())
                .offset(offset)
                .limit(page_size)
                .all())

        items = []
        for i, t in enumerate(rows, start=1+offset):
            items.append(TicketListItem(
                n=i,
                ticket_id=t.id,
                purchaser_name=t.purchaser_name,
                event_id=t.event_id,
                national_id=t.national_id,
                phone=t.phone,
                event_date=t.event_date,
                event_at=_utc(t.event_at) if t.event_at else _utc(datetime.combine(t.event_date, time(20,0)).replace(tzinfo=timezone.utc)),
                created_at=_utc(t.created_at),
                expires_at=_utc(t.expires_at),
                used_at=_utc(t.used_at),
                is_used=t.is_used,
                created_by=t.created_by,
                validated_by=t.validated_by,
            ))

        return TicketListResponse(
            items=items,
            total=total,
            used=used_count,
            unused=unused_count,
            page=page,
            page_size=page_size,
        )
    finally:
        db.close()

@app.get("/tickets/{ticket_id}", response_model=TicketStatusResponse, tags=["tickets"])
def get_ticket_status(ticket_id: str, _=Depends(require_perm("tickets:read_status"))):
    db = SessionLocal()
    try:
        t: Ticket = db.get(Ticket, ticket_id)
        if not t:
            raise HTTPException(status_code=404, detail="Ticket no encontrado")

        now = datetime.now(timezone.utc)
        exp_dt = _utc(t.expires_at)
        valid = (not t.is_used) and (exp_dt is None or now <= exp_dt)

        return TicketStatusResponse(
            ticket_id=t.id,
            valid=valid,
            is_used=t.is_used,
            purchaser_name=t.purchaser_name,
            event_id=t.event_id,
            national_id=t.national_id,
            phone=t.phone,
            event_date=t.event_date,
            event_at=_utc(t.event_at) if t.event_at else _utc(datetime.combine(t.event_date, time(20,0)).replace(tzinfo=timezone.utc)),
            created_at=_utc(t.created_at),
            expires_at=exp_dt,
            used_at=_utc(t.used_at),
        )
    finally:
        db.close()
