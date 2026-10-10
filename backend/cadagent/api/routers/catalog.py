"""bd_warehouse standard-component catalog (types + sizes)."""

from typing import Any

from fastapi import APIRouter, HTTPException

from cadagent.cad import catalog

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("")
async def get_catalog(category: str | None = None) -> dict[str, Any]:
    cat = catalog.build_catalog()
    if category:
        if category not in cat:
            raise HTTPException(404, f"unknown category '{category}'")
        return {category: cat[category]}
    return cat


@router.get("/{category}/{class_name}")
async def get_class(category: str, class_name: str) -> dict[str, Any]:
    cat = catalog.build_catalog()
    entry = (cat.get(category) or {}).get(class_name)
    if entry is None:
        raise HTTPException(404, "catalog entry not found")
    return {class_name: entry}
