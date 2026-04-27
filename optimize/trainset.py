"""
optimize/trainset.py
--------------------
Dataset de entrenamiento y validación para la optimización GEPA.

Contiene ejemplos de pares (documento_normativo → rúbrica_ideal).
Los ejemplos son representativos del dominio: normativas académicas,
reglamentos institucionales, y estándares de cumplimiento.

Para agregar tus propios ejemplos:
  1. Agrega un dict a EJEMPLOS_CRUDOS con las claves:
     - "documento": texto del PDF (puede ser resumido)
     - "ontologia": resumen de entidades (puede estar vacío "")
     - "rubrica": rúbrica ideal de referencia
  2. O cargalos desde archivos JSON con load_from_json()
"""

import json
import random
from pathlib import Path
import dspy

# ---------------------------------------------------------------------------
# Ejemplos de referencia
# ---------------------------------------------------------------------------

EJEMPLOS_CRUDOS = [

    # ----- Ejemplo 1: Normativa de trabajos académicos ----------------------
    {
        "documento": """
NORMATIVA DE CALIDAD PARA TRABAJOS ACADÉMICOS — Universidad Nacional

Artículo 1. Estructura obligatoria
Todo trabajo académico deberá incluir: portada con datos completos del autor,
resumen de 150-250 palabras, introducción con planteamiento del problema,
marco teórico con referencias bibliográficas actualizadas (últimos 5 años),
metodología detallada, resultados con tablas o figuras, conclusiones
y bibliografía en formato APA 7ma edición.

Artículo 2. Extensión
- Trabajos de grado: mínimo 40 páginas sin apéndices.
- Tesis de posgrado: mínimo 80 páginas sin apéndices.
- Artículos de investigación: entre 6.000 y 10.000 palabras.

Artículo 3. Citación y plagio
El porcentaje de similitud detectado por sistemas anti-plagio no deberá
superar el 20% del total del documento. Toda cita directa debe ir entre
comillas y con referencia en el texto.

Artículo 4. Presentación formal
Fuente: Times New Roman 12 o Arial 11. Interlineado: 1.5. Márgenes: 2.5 cm
en todos los lados. Numeración de páginas desde la introducción.
        """,
        "ontologia": "entidades: estructura_trabajo, extension_paginas, citacion_apa, formato_tipografico | relaciones: estructura_trabajo REQUIERE citacion_apa",
        "rubrica": """• Institución: Universidad Nacional
• Ámbito de Aplicación: Trabajos académicos de grado y posgrado
• Normativa de Referencia: Normativa de Calidad para Trabajos Académicos
• Nivel de Criticidad: Medio
• Objetivos de la evaluación: Verificar que el trabajo académico cumpla con los requisitos formales, estructurales y de citación establecidos por la normativa institucional

| Área de Cumplimiento | Criterio de evaluación | Evidencias observables | Nivel mínimo aprobatorio |
|---|---|---|---|
| Estructura | El trabajo incluye todas las secciones obligatorias según el Art. 1 | Portada, resumen, introducción, marco teórico, metodología, resultados, conclusiones y bibliografía presentes | Todas las secciones presentes y desarrolladas (ej: resumen de 180 palabras con planteamiento, metodología y hallazgos) |
| Extensión | El trabajo cumple con la extensión mínima según su tipo (Art. 2) | Conteo de páginas sin apéndices verificado | Al menos 40 páginas para grado, 80 para posgrado (ej: tesis de maestría con 92 páginas de contenido) |
| Citación | El porcentaje de similitud no supera el 20% (Art. 3) | Reporte del sistema anti-plagio adjunto | Reporte con índice ≤ 20% (ej: informe Turnitin con 14% de similitud) |
| Citación | Todas las citas directas van entre comillas con referencia en texto (Art. 3) | Revisión de al menos 5 citas directas en el documento | Citas entre comillas con autor, año y página (ej: "la educación es transformadora" (Freire, 1970, p. 43)) |
| Formato | El trabajo usa la tipografía y márgenes especificados (Art. 4) | Verificación en propiedades del documento Word/PDF | Times New Roman 12 o Arial 11, interlineado 1.5, márgenes 2.5 cm en todos los lados (ej: documento con configuración de página visible en metadatos) |
| Bibliografía | Las referencias bibliográficas están en formato APA 7ma edición (Art. 1) | Al menos el 80% de las referencias verificadas | Referencias con autor, año, título, editorial y DOI cuando corresponda (ej: García, M. (2021). Metodología de la investigación. Pearson.) |
"""
    },

    # ----- Ejemplo 2: Reglamento de uso de laboratorio ----------------------
    {
        "documento": """
REGLAMENTO DE USO DE LABORATORIO DE INFORMÁTICA — Facultad de Ingeniería

Capítulo I: Acceso y Uso
Art. 1 — Los estudiantes podrán acceder al laboratorio en el horario establecido
(lunes a viernes de 8:00 a 20:00 hs) únicamente con credencial universitaria vigente.
Art. 2 — Cada sesión de uso tiene una duración máxima de 2 horas continuas cuando
hay lista de espera. El responsable del laboratorio llevará registro de entradas y salidas.
Art. 3 — Está prohibido consumir alimentos o bebidas dentro del laboratorio.
Art. 4 — Los equipos deben apagarse correctamente al finalizar la sesión. No está
permitido instalar software sin autorización del responsable técnico.

Capítulo II: Responsabilidades
Art. 5 — El estudiante es responsable de los daños causados por uso indebido del equipo.
Art. 6 — Cualquier falla técnica debe reportarse inmediatamente al responsable presente.
Art. 7 — El laboratorio cuenta con cámaras de seguridad. Las grabaciones se conservan
por 30 días.

Capítulo III: Sanciones
Art. 8 — El incumplimiento de este reglamento puede derivar en suspensión temporal
(1-5 días) o permanente del acceso, según la gravedad de la infracción.
        """,
        "ontologia": "entidades: acceso_laboratorio, horario_uso, credencial_universitaria, responsable_tecnico, sanciones | relaciones: acceso_laboratorio REQUIERE credencial_universitaria | incumplimiento DERIVA_EN sanciones",
        "rubrica": """• Institución: Facultad de Ingeniería
• Ámbito de Aplicación: Uso del Laboratorio de Informática por estudiantes
• Normativa de Referencia: Reglamento de Uso de Laboratorio de Informática
• Nivel de Criticidad: Medio
• Objetivos de la evaluación: Verificar que los estudiantes y el personal responsable cumplan con las normas de acceso, uso y seguridad del laboratorio de informática

| Área de Cumplimiento | Criterio de evaluación | Evidencias observables | Nivel mínimo aprobatorio |
|---|---|---|---|
| Acceso | El estudiante accede al laboratorio con credencial universitaria vigente en el horario permitido (Art. 1) | Registro de acceso con credencial, horario de entrada y salida | Credencial vigente presentada en cada acceso dentro del horario de 8:00 a 20:00 hs (ej: acceso registrado el 15/03 a las 10:30 con credencial #4521) |
| Control de sesión | Las sesiones no superan las 2 horas cuando hay lista de espera (Art. 2) | Registro del responsable con hora de inicio y fin de cada sesión | Registro de sesión con horario completo firmado por el responsable (ej: sesión 09:00-10:50, lista de espera de 3 personas verificada) |
| Normas de uso | No se consumen alimentos ni bebidas dentro del laboratorio (Art. 3) | Ausencia de evidencia de consumo; supervisión del responsable | Cero incidentes registrados por consumo de alimentos durante el período evaluado (ej: bitácora del mes sin observaciones al respecto) |
| Normas de uso | Los equipos se apagan correctamente y no se instala software no autorizado (Art. 4) | Verificación al final de cada sesión; log del sistema | Equipos en estado correcto al cierre; sin software no autorizado detectado (ej: revisión semanal del técnico sin hallazgos) |
| Responsabilidad | Las fallas técnicas se reportan inmediatamente al responsable (Art. 6) | Registro de reportes de incidentes técnicos | Al menos 1 reporte formal por incidente con fecha, hora y descripción (ej: reporte firmado de teclado roto el 22/03 a las 14:15) |
| Seguridad | Las grabaciones de cámaras se conservan por 30 días (Art. 7) | Verificación del sistema de grabación y política de retención | Sistema configurado con retención de 30 días; verificable en la configuración del servidor de seguridad (ej: captura de pantalla de configuración del NVR) |
"""
    },

    # ----- Ejemplo 3: Política de teletrabajo --------------------------------
    {
        "documento": """
POLÍTICA DE TELETRABAJO — Recursos Humanos

1. Elegibilidad
Podrán acceder al teletrabajo los empleados con al menos 6 meses de antigüedad,
calificación "Satisfactorio" o superior en la última evaluación de desempeño,
y cuyas funciones sean compatibles con el trabajo remoto según el área.

2. Modalidades
- Teletrabajo total: 100% remoto, con presencia obligatoria en la oficina
  al menos 1 vez por mes para reuniones de equipo.
- Teletrabajo parcial: 2 a 3 días por semana remotos, el resto presencial.
La modalidad se define por acuerdo entre el empleado y su jefe inmediato,
con aprobación de RRHH.

3. Equipamiento y conectividad
El empleado debe contar con conexión a internet de al menos 10 Mbps y
computadora compatible con las herramientas corporativas. La empresa proveerá
VPN y licencias de software. Los daños al equipamiento propio no son
responsabilidad de la empresa.

4. Disponibilidad y comunicación
El empleado deberá estar disponible en el horario laboral definido (9:00 a 18:00 hs),
responder comunicaciones dentro de los 30 minutos en horario laboral,
y asistir puntualmente a todas las reuniones virtuales agendadas.

5. Seguridad de la información
Está prohibido usar redes WiFi públicas sin VPN activa. Los documentos
confidenciales no deben descargarse en dispositivos personales sin autorización.
        """,
        "ontologia": "entidades: elegibilidad_teletrabajo, modalidad_teletrabajo, equipamiento, disponibilidad_laboral, seguridad_informacion | relaciones: teletrabajo REQUIERE elegibilidad | modalidad DEFINE_A disponibilidad",
        "rubrica": """• Institución: Área de Recursos Humanos
• Ámbito de Aplicación: Empleados en modalidad de teletrabajo (total o parcial)
• Normativa de Referencia: Política de Teletrabajo — Recursos Humanos
• Nivel de Criticidad: Medio
• Objetivos de la evaluación: Verificar que los empleados y sus jefes directos cumplan con los requisitos de elegibilidad, modalidad, conectividad, disponibilidad y seguridad establecidos en la política de teletrabajo

| Área de Cumplimiento | Criterio de evaluación | Evidencias observables | Nivel mínimo aprobatorio |
|---|---|---|---|
| Elegibilidad | El empleado cumple los requisitos de antigüedad y desempeño para acceder al teletrabajo (Sección 1) | Legajo con fecha de ingreso y última evaluación de desempeño | Al menos 6 meses de antigüedad y calificación "Satisfactorio" o superior en la última evaluación (ej: empleado con ingreso 01/07/2023 y evaluación "Muy Bueno" en diciembre 2023) |
| Modalidad | La modalidad de teletrabajo está acordada y aprobada formalmente (Sección 2) | Acuerdo firmado por el empleado, jefe inmediato y RRHH | Documento de acuerdo de modalidad firmado por las 3 partes con fecha y modalidad especificada (ej: acuerdo de teletrabajo parcial 3 días remotos firmado el 15/01/2024) |
| Presencia obligatoria | En modalidad total, el empleado asiste a la oficina al menos 1 vez por mes (Sección 2) | Registro de asistencia mensual a reuniones de equipo presenciales | Al menos 1 asistencia presencial registrada por mes (ej: firma en planilla de asistencia del 18/03/2024) |
| Conectividad | El empleado cuenta con conexión de al menos 10 Mbps y equipamiento compatible (Sección 3) | Test de velocidad de internet y checklist de equipamiento aprobado | Test de velocidad ≥ 10 Mbps documentado y equipamiento verificado por TI (ej: reporte de speed test con 25 Mbps y aprobación técnica fechada) |
| Disponibilidad | El empleado responde comunicaciones dentro de los 30 minutos en horario laboral (Sección 4) | Registros de tiempos de respuesta en plataforma corporativa | Tiempo de respuesta promedio ≤ 30 minutos verificable en logs del sistema de mensajería (ej: promedio de 18 minutos en el último mes según reporte de Teams) |
| Seguridad | No se usan redes WiFi públicas sin VPN activa ni se descargan documentos confidenciales en dispositivos personales (Sección 5) | Logs de VPN y política de DLP aplicada | Conexión VPN activa en todos los accesos remotos registrados (ej: log de conexiones VPN del mes sin accesos desde redes no corporativas sin VPN) |
"""
    },

    # ----- Ejemplo 4: Normativa de seguridad e higiene ----------------------
    {
        "documento": """
REGLAMENTO INTERNO DE SEGURIDAD E HIGIENE EN EL TRABAJO

Sección A — Elementos de Protección Personal (EPP)
Todo el personal operativo debe usar casco, guantes y calzado de seguridad
en el área de producción. El uso de chaleco reflectivo es obligatorio en el
depósito y estacionamiento. Los EPP serán provistos por la empresa y deben
renovarse ante deterioro evidente.

Sección B — Capacitación
El personal debe recibir capacitación en seguridad e higiene al ingresar y
una actualización anual mínima de 4 horas. Los registros de capacitación
(listas de asistencia firmadas) deben archivarse por 5 años.

Sección C — Incidentes y accidentes
Todo incidente o accidente debe reportarse dentro de las 24 horas al área
de Seguridad, completando el formulario F-SH-01. Los accidentes con días
de baja laboral deben comunicarse adicionalmente a la ART en el plazo legal.

Sección D — Inspecciones
El área de Seguridad e Higiene realizará inspecciones mensuales de las
instalaciones. Los hallazgos deben documentarse en el informe de inspección
y los incumplimientos subsanarse dentro de los 15 días hábiles.
        """,
        "ontologia": "entidades: EPP, capacitacion_seguridad, reporte_incidente, inspeccion_mensual | relaciones: personal_operativo REQUIERE EPP | incidente REQUIERE reporte_24hs",
        "rubrica": """• Institución: Área de Seguridad e Higiene
• Ámbito de Aplicación: Personal operativo y de depósito de la empresa
• Normativa de Referencia: Reglamento Interno de Seguridad e Higiene en el Trabajo
• Nivel de Criticidad: Alto
• Objetivos de la evaluación: Verificar el cumplimiento de las obligaciones de protección personal, capacitación, reporte de incidentes e inspecciones establecidas en el reglamento de seguridad e higiene

| Área de Cumplimiento | Criterio de evaluación | Evidencias observables | Nivel mínimo aprobatorio |
|---|---|---|---|
| Elementos de Protección Personal | El personal operativo usa casco, guantes y calzado de seguridad en el área de producción (Sección A) | Inspección visual y registro fotográfico durante el turno | 100% del personal en área de producción con EPP completo durante la inspección (ej: fotografías fechadas del 20/03 con todo el turno mañana con EPP) |
| Elementos de Protección Personal | Se usa chaleco reflectivo en depósito y estacionamiento; los EPP deteriorados se renuevan (Sección A) | Verificación de estado de EPPs; registro de entregas de nuevos EPPs | Chalecos en buen estado en uso; registro de renovación cuando corresponde (ej: acta de entrega de 5 chalecos nuevos el 10/03 por deterioro documentado) |
| Capacitación | El personal recibe capacitación en seguridad al ingresar y actualización anual de mínimo 4 horas (Sección B) | Listas de asistencia firmadas a capacitaciones; certificados si aplica | Lista firmada de capacitación de ingreso y al menos 4 horas anuales por persona (ej: 8 empleados con firma en capacitación del 15/02 de 4 horas) |
| Capacitación | Los registros de capacitación se archivan por 5 años (Sección B) | Carpeta física o digital con listas de asistencia históricas | Registros de los últimos 5 años disponibles y ordenados cronológicamente (ej: carpeta con actas desde 2020 hasta la fecha, foliadas) |
| Reporte de incidentes | Todo incidente se reporta dentro de las 24 horas usando el formulario F-SH-01 (Sección C) | Formularios F-SH-01 completados con fecha y hora de ocurrencia y reporte | Formulario presentado dentro de las 24 horas del incidente con todos los campos completos (ej: incidente del 05/04 a las 14:00 reportado el 06/04 a las 08:00) |
| Inspecciones | Se realizan inspecciones mensuales y los incumplimientos se subsanan en 15 días hábiles (Sección D) | Informes de inspección mensuales; plan de acción con fechas de cierre | Al menos 1 informe de inspección por mes con hallazgos y fechas de cierre ≤ 15 días hábiles (ej: informe de marzo con 2 hallazgos cerrados el 18/03 y 22/03) |
"""
    },

    # ----- Ejemplo 5: Reglamento de becas ------------------------------------
    {
        "documento": """
REGLAMENTO DE BECAS ESTUDIANTILES — Secretaría de Bienestar Universitario

Capítulo 1: Tipos de becas
La universidad otorga tres tipos de becas:
a) Beca de Apoyo Económico: cubre el 50% del arancel mensual.
b) Beca de Excelencia Académica: cubre el 100% del arancel mensual.
c) Beca de Materiales: suma fija de $15.000 por cuatrimestre para compra de materiales.

Capítulo 2: Requisitos generales
Para acceder a cualquier tipo de beca, el/la estudiante debe:
- Estar inscripto/a en una carrera de grado de la universidad.
- Haber aprobado al menos el 60% de las materias del año anterior.
- No tener materias aplazadas pendientes de regularización.
- Presentar la documentación requerida antes del 31 de marzo de cada año.

Capítulo 3: Mantenimiento de la beca
El/la becario/a debe mantener un promedio mínimo de 7 puntos (escala 1-10)
y no reprobar más de 2 materias por año académico. El incumplimiento implica
la suspensión automática de la beca hasta regularizar la situación.

Capítulo 4: Renovación
La renovación es automática si se cumplen los requisitos de mantenimiento.
De lo contrario, se debe presentar una solicitud justificada ante la Secretaría
antes del 15 de febrero.
        """,
        "ontologia": "entidades: tipos_beca, requisitos_ingreso, mantenimiento_beca, renovacion_beca | relaciones: beca REQUIERE requisitos_ingreso | mantenimiento HABILITA renovacion",
        "rubrica": """• Institución: Secretaría de Bienestar Universitario
• Ámbito de Aplicación: Estudiantes de grado solicitantes y beneficiarios/as de becas universitarias
• Normativa de Referencia: Reglamento de Becas Estudiantiles — Secretaría de Bienestar Universitario
• Nivel de Criticidad: Medio
• Objetivos de la evaluación: Verificar el cumplimiento de los requisitos de acceso, mantenimiento y renovación de becas estudiantiles conforme al reglamento vigente

| Área de Cumplimiento | Criterio de evaluación | Evidencias observables | Nivel mínimo aprobatorio |
|---|---|---|---|
| Elegibilidad | La persona solicitante está inscripta en una carrera de grado y aprobó al menos el 60% de las materias del año anterior (Cap. 2) | Certificado de inscripción vigente y certificado analítico del año anterior | Inscripción activa en carrera de grado con al menos 60% de materias aprobadas (ej: analítico con 8 de 12 materias aprobadas = 67%) |
| Elegibilidad | No hay materias aplazadas pendientes de regularización al momento de la solicitud (Cap. 2) | Certificado analítico actualizado sin aplazos pendientes | Cero aplazos pendientes de regularización en el certificado analítico de solicitud (ej: analítico del 20/03 sin materias en rojo) |
| Documentación | La documentación requerida se presenta antes del 31 de marzo (Cap. 2) | Sello de recepción o acuse de recibo electrónico con fecha | Documentación completa recibida con fecha ≤ 31 de marzo (ej: acuse de recibo por sistema el 28/03 a las 16:42) |
| Mantenimiento | La persona becaria mantiene promedio mínimo de 7 puntos durante el año académico (Cap. 3) | Certificado analítico con promedio general calculado | Promedio general ≥ 7.0 puntos en el analítico del período evaluado (ej: promedio de 7.8 con 6 materias aprobadas en el año) |
| Mantenimiento | No se reprueban más de 2 materias en el año académico (Cap. 3) | Certificado analítico del período | Máximo 2 materias reprobadas en el año académico (ej: analítico con 1 materia reprobada en el primer cuatrimestre) |
| Renovación | Si no se cumplen los requisitos de mantenimiento, se presenta solicitud justificada antes del 15 de febrero (Cap. 4) | Nota de solicitud de renovación con justificación y sello de recepción | Solicitud presentada con justificación documentada antes del 15/02 (ej: nota presentada el 10/02 con certificado médico adjunto por baja de promedio) |
"""
    },
]


# ---------------------------------------------------------------------------
# Función para construir los dspy.Example a partir de los crudos
# ---------------------------------------------------------------------------

def _build_examples(datos: list[dict]) -> list[dspy.Example]:
    """Convierte los dicts crudos en dspy.Example con inputs declarados."""
    ejemplos = []
    for d in datos:
        ejemplo = dspy.Example(
            documento_texto=d["documento"].strip(),
            ontologia_resumen=d["ontologia"].strip(),
            rubrica=d["rubrica"].strip(),
        ).with_inputs("documento_texto", "ontologia_resumen")
        ejemplos.append(ejemplo)
    return ejemplos


def get_trainset(
    train_ratio: float = 0.7,
    seed: int = 42,
    extra_ejemplos: list[dict] | None = None,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """
    Retorna (trainset, valset) mezclados y divididos.

    Args:
        train_ratio: Proporción del total para entrenamiento (default 0.7 = 70%).
        seed: Semilla para reproducibilidad.
        extra_ejemplos: Lista adicional de dicts con claves documento/ontologia/rubrica.

    Returns:
        Tupla (trainset, valset) de dspy.Example.
    """
    datos = EJEMPLOS_CRUDOS.copy()
    if extra_ejemplos:
        datos.extend(extra_ejemplos)

    ejemplos = _build_examples(datos)

    rng = random.Random(seed)
    rng.shuffle(ejemplos)

    split = max(1, int(len(ejemplos) * train_ratio))
    trainset = ejemplos[:split]
    valset = ejemplos[split:] or ejemplos[-1:]  # mínimo 1 ejemplo en val

    return trainset, valset


def load_from_json(path: str | Path) -> list[dspy.Example]:
    """
    Carga ejemplos adicionales desde un archivo JSON.

    Formato esperado del JSON:
    [
        {
            "documento": "texto del documento...",
            "ontologia": "resumen de ontología...",
            "rubrica": "rúbrica ideal en Markdown..."
        },
        ...
    ]
    """
    with open(path, encoding="utf-8") as f:
        datos = json.load(f)
    return _build_examples(datos)


# ---------------------------------------------------------------------------
# Diagnóstico rápido al ejecutar el módulo directamente
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train, val = get_trainset()
    print(f"✅ Trainset: {len(train)} ejemplos")
    print(f"✅ Valset:   {len(val)} ejemplos")
    print("\nEjemplo de muestra (trainset[0]):")
    print(f"  documento ({len(train[0].documento_texto)} chars)")
    print(f"  rubrica   ({len(train[0].rubrica)} chars)")
