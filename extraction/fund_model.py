from pydantic import BaseModel, Field
from typing import Optional


class FundModel(BaseModel):
    """Modelo de datos estructurado para un fondo de inversión extraído de su DFI."""

    # ── Identificación ──────────────────────────────────────────────
    nombre_fondo: str = Field(description="Nombre completo del fondo de inversión")
    isin: str = Field(description="Código ISIN único del fondo")
    numero_registro: Optional[str] = Field(None, description="Número de registro oficial (CNMV u otro organismo)")
    gestora: str = Field(description="Nombre de la sociedad gestora del fondo")

    # ── Clasificación ───────────────────────────────────────────────
    categoria: Optional[str] = Field(None, description="Categoría del fondo: Renta Variable, Renta Fija, Mixto, Monetario, Global, etc.")
    tipo_gestion: Optional[str] = Field(None, description="Tipo de gestión: 'activa' o 'pasiva'")
    indice_referencia: Optional[str] = Field(None, description="Benchmark o índice de referencia utilizado")
    universo_activos: Optional[list[str]] = Field(None, description="Lista de activos en los que invierte: acciones, bonos, derivados, otros fondos, depósitos, etc.")
    politica_inversion: Optional[str] = Field(None, description="Descripción completa de la política de inversión del fondo. Campo principal para vectorización RAG.")

    # ── Riesgo ──────────────────────────────────────────────────────
    nivel_riesgo: Optional[int] = Field(None, ge=1, le=7, description="Nivel de riesgo en escala 1 (menor) a 7 (mayor). Buscar: indicador numérico 1-7 en el SRRI/SRI, número resaltado en la escala de riesgo, 'Indicador de riesgo: X', 'clase de riesgo X', o la posición del cuadro marcado en el gráfico de barras 1-7")
    perfil_riesgo: Optional[str] = Field(None, description="Perfil de riesgo cualitativo: Bajo, Medio o Alto")
    perfil_inversor: Optional[str] = Field(None, description="Perfil del inversor objetivo: Conservador, Moderado o Decidido")
    volatilidad: Optional[float] = Field(None, description="Volatilidad histórica o estimada del fondo expresada en porcentaje")

    # ── Plazos y liquidez ───────────────────────────────────────────
    horizonte_recomendado_anios: Optional[int] = Field(None, description="Horizonte temporal mínimo recomendado expresado en años")
    restricciones_liquidez: Optional[str] = Field(None, description="Condiciones o restricciones para el reembolso o bloqueo del capital")

    # ── Capital ─────────────────────────────────────────────────────
    importe_minimo_inversion: Optional[float] = Field(None, description="Capital mínimo requerido para suscribir el fondo, en euros")

    # ── Comisiones ──────────────────────────────────────────────────
    comision_suscripcion: Optional[float] = Field(None, description="Comisión de entrada o suscripción en porcentaje")
    comision_reembolso: Optional[float] = Field(None, description="Comisión de salida o reembolso en porcentaje")
    comision_gestion: Optional[float] = Field(None, description="Comisión anual de gestión en porcentaje")
    comision_exito: Optional[float] = Field(None, description="Comisión de éxito sobre rendimientos positivos en porcentaje")
    comision_deposito: Optional[float] = Field(None, description="Comisión de depósito o custodia en porcentaje")

    # ── Distribución ────────────────────────────────────────────────
    distribucion_sectorial: Optional[dict[str, float]] = Field(None, description="Distribución porcentual por sectores económicos, ej: {'tecnología': 30.5, 'salud': 20.0}")
    distribucion_geografica: Optional[dict[str, float]] = Field(None, description="Distribución porcentual por zonas geográficas, ej: {'Europa': 60.0, 'EEUU': 30.0}")
    distribucion_renta_variable_pct: Optional[float] = Field(None, description="Porcentaje máximo o actual invertido en renta variable")
    distribucion_renta_fija_pct: Optional[float] = Field(None, description="Porcentaje máximo o actual invertido en renta fija")
    calidad_crediticia: Optional[str] = Field(None, description="Para fondos de deuda: Investment Grade, High Yield o Mixto")

    # ── Características adicionales ─────────────────────────────────
    divisa_cobertura: Optional[bool] = Field(None, description="True si el fondo tiene cobertura de riesgo divisa, False si no la tiene")
    politica_dividendos: Optional[str] = Field(None, description="Política de dividendos: 'acumulación' (reinversión) o 'distribución' (pago en efectivo)")
    esg: Optional[bool] = Field(None, description="True si el fondo cumple criterios de sostenibilidad ESG, False si no los cumple")

    # ── Metadatos ───────────────────────────────────────────────────
    pdf_origen: Optional[str] = Field(None, description="Nombre o ruta del archivo PDF fuente del que se extrajo la información")
