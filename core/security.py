from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from core.config import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="v1/auth/login")


async def get_current_user_id(token: str = Depends(oauth2_scheme)):
    """
    Validates JWT token using OAuth2PasswordBearer.
    FastAPI will automatically look for the 'Authorization: Bearer <token>' header.
    """

    try:
        token = token.strip()
        # 3. Decode the token using values from .env via settings
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
            options={"verify_signature": False, "verify_exp": True},
        )
        print(f"DEBUG - Payload: {payload}")

        if not payload:
            raise HTTPException(status_code=401, detail="Token payload is empty")
        # 4. Extract 'userId' as shown in your portal's local storage
        user_id = payload.get("userId")

        if not user_id:
            # Fallback: check if the key is 'id' or 'sub' instead of 'userId'
            user_id = payload.get("sub") or payload.get("id")

        return user_id

    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Authentication failed: Token is invalid or has expired.",
        )
