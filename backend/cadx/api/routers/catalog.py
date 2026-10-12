"""bd_warehouse standard-component catalog (types + sizes + lookups)."""

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from cadx.cad import catalog

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("")
async def get_catalog(category: str | None = None) -> dict[str, Any]:
    cat = catalog.build_catalog()
    if category:
        if category not in cat:
            raise HTTPException(404, f"unknown category '{category}'")
        return {category: cat[category]}
    return cat


@router.get("/search")
async def search_catalog(q: str = Query("")) -> dict[str, Any]:
    return {"results": catalog.search(q)}


@router.get("/lookup")
async def lookup(
    class_name: str = Query(...),
    fastener_type: str | None = Query(None),
    size: str | None = Query(None),
    length: float | None = Query(None),
) -> dict[str, Any]:
    """Instantiate a part and return its real dimensions (hole tables etc.)."""
    try:
        return catalog.lookup_part(
            class_name, fastener_type=fastener_type, size=size, length=length
        )
    except ValueError as e:
        raise HTTPException(404, str(e)) from e


@router.get("/{category}/{class_name}")
async def get_class(category: str, class_name: str) -> dict[str, Any]:
    cat = catalog.build_catalog()
    entry = (cat.get(category) or {}).get(class_name)
    if entry is None:
        raise HTTPException(404, "catalog entry not found")
    return {class_name: entry}
