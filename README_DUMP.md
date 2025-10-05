# 🧩 Endpoint `/__admin__/dump-postgres`

## ⚠️ Propósito y advertencia

Este endpoint **solo debe usarse en entorno local o de desarrollo**.  
Su función es realizar un **backup completo** de la base de datos PostgreSQL configurada en `DATABASE_URL`, utilizando el comando `pg_dump` en formato **custom (`-Fc`)**, y devolver el archivo `.dump` como descarga directa.

> ⚠️ **Nunca habilites este endpoint en producción.**  
> Permite extraer toda la base de datos y requiere permisos de administrador (`admin:dump`).

---

## 🧰 Requisitos previos

### 1️⃣ Tener instalado **PostgreSQL Client Tools**

Debes contar con `pg_dump`, `psql`, `pg_restore` y `createdb` en tu PATH.

#### 🔧 Instalación en Windows
Descarga desde el sitio oficial de PostgreSQL:  
👉 [https://www.postgresql.org/download/windows/](https://www.postgresql.org/download/windows/)

Durante la instalación, marca la opción **Command Line Tools**.

Por defecto, los binarios quedan en:
```
C:\Program Files\PostgreSQL\18\bin
```

---

### 2️⃣ Agregar `pg_dump` al PATH del sistema

#### Opción A — Temporal (solo para la sesión actual)
```bat
setx PATH "%PATH%;C:\Program Files\PostgreSQL\18\bin"
```

#### Opción B — Permanente (recomendado)
1. Abre **Panel de control → Sistema → Configuración avanzada del sistema → Variables de entorno**  
2. Edita la variable **PATH** del sistema.  
3. Agrega esta ruta:
   ```
   C:\Program Files\PostgreSQL\18\bin
   ```
4. **Reinicia el equipo o la terminal** para aplicar los cambios.

---

### 3️⃣ Configurar variables de entorno

En tu archivo `.env` local:
```env
DATABASE_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/tickets_local
SECRET_KEY=algo_secreto
MIGRATE_DATA=false
```

**Notas:**
- Usuario y contraseña por defecto: `postgres:postgres`
- Usa `127.0.0.1` como host y puerto `5432`
- No incluyas `sslmode=require` en local

---

### 4️⃣ Verificar instalación

Ejecuta:
```bat
pg_dump --version
```

Si ves algo como:
```
pg_dump (PostgreSQL) 18.0
```
✅ Todo está correcto.

---

## 🧪 Uso del endpoint

Con la aplicación corriendo localmente:
```bash
uvicorn main:app --reload
```

Accede a Swagger:
```
http://127.0.0.1:8000/docs
```

Busca el endpoint:
```
GET /__admin__/dump-postgres
```

Autentícate con un usuario que tenga el permiso `admin:dump`.

Al ejecutarlo, descargará un archivo con nombre similar a:
```
dump_tickets_20251005T042450Z.dump
```

---

## 💾 Restaurar el dump en local

### 1️⃣ Crear una base vacía

```bat
createdb -U postgres -h 127.0.0.1 tickets_local
```

Si aparece un error de *collation mismatch*, puedes refrescarla:

```sql
ALTER DATABASE template1 REFRESH COLLATION VERSION;
ALTER DATABASE postgres REFRESH COLLATION VERSION;
```

O crearla desde `template0`:
```bat
createdb -U postgres -h 127.0.0.1 -T template0 tickets_local
```

---

### 2️⃣ Restaurar el dump

```bat
pg_restore -U postgres -h 127.0.0.1 -d tickets_local --no-owner --no-privileges dump_tickets_20251005T042450Z.dump
```

> 💡 Usa `--no-owner` y `--no-privileges` para evitar errores de permisos.

---

### 3️⃣ Verificar la restauración

Conéctate al servidor local:
```bat
"C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -d tickets_local -h 127.0.0.1
```

Dentro de `psql`:
```sql
\encoding UTF8           -- asegúrate de usar UTF8
\dt                      -- lista las tablas
SELECT COUNT(*) FROM tickets;
SELECT * FROM tickets LIMIT 5;
```

Si ves tus datos, el dump se restauró correctamente 🎉

---

## 🌍 Configurar codificación UTF-8 en consola

Para evitar problemas con acentos o caracteres especiales en Windows:

1. En tu terminal (CMD o PowerShell):
   ```bat
   chcp 65001
   ```
   Esto cambia la página de código a UTF-8.

2. Dentro de `psql`:
   ```sql
   \encoding UTF8
   ```

---

## 🧩 Resumen de pasos

| Paso | Descripción | Comando |
|------|--------------|---------|
| 1️⃣ | Instalar PostgreSQL tools | `pg_dump --version` |
| 2️⃣ | Agregar al PATH | `setx PATH "%PATH%;C:\Program Files\PostgreSQL\18\bin"` |
| 3️⃣ | Reiniciar terminal | *(para aplicar variables)* |
| 4️⃣ | Ejecutar app localmente | `uvicorn main:app --reload` |
| 5️⃣ | Descargar dump desde Swagger | `/__admin__/dump-postgres` |
| 6️⃣ | Crear base vacía | `createdb -U postgres tickets_local` |
| 7️⃣ | Restaurar dump | `pg_restore -U postgres -d tickets_local --no-owner --no-privileges archivo.dump` |
| 8️⃣ | Verificar datos | `psql -U postgres -d tickets_local -h 127.0.0.1` |
| 9️⃣ | Establecer UTF-8 | `chcp 65001` y `\encoding UTF8` |

---

## 🧠 Notas finales

- Usa este endpoint **solo en entornos controlados o locales**.  
- En producción, la base de datos debe respaldarse mediante un sistema externo (Railway, RDS, etc).  
- Puedes automatizar el proceso con un script PowerShell que ejecute `pg_dump` y `pg_restore`.

---

✳️ **Autor:** Equipo Backend  
📅 **Última actualización:** Octubre 2025  
🔒 **Nivel de acceso:** Administrador (`admin:dump`)
