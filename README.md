
---

### 📄 research_note.md (documento de hallazgo)

```markdown
# Nota técnica: Cómo resolver el olvido catastrófico en adaptación de expertos identitarios

**Autora:** Magali Ofelia Gafe  
**Fecha:** 11 de mayo de 2026  
**Contexto:** Proyecto Identity Experts MoE

## 1. El problema

En sistemas de Mixture of Experts (MoE) donde cada experto se entrena en un silo de datos homogéneo (ej. productos, parcelas, experimentos), surge la necesidad de **actualizar un experto** cuando su dominio cambia gradualmente (ej. variación numérica en la relación entrada-salida). La intuición sugiere que un cambio pequeño debería poderse corregir con fine-tuning.

Sin embargo, experimentos previos con **EDE** (Ecuación Diferencial Estocástica), **EWC** (Elastic Weight Consolidation), buffer de replay e incluso **gradiente natural** (geodésica negativa) mostraron un patrón consistente:

- **Olvido significativo** en el dominio original (hasta 75%).
- **Ninguna mejora** en el nuevo dominio (incluso empeoraba ligeramente).

Esto sugería que no existía un punto de compromiso en el espacio de parámetros que fuera bueno para ambos dominios.

## 2. La solución inspirada en Tencent Hy3

Los modelos recientes de Tencent (Hy3, Hunyuan-Large) utilizan una arquitectura con **un experto compartido** y **múltiples expertos especializados**. El experto compartido captura el conocimiento común a todos los dominios, mientras que los especializados se encargan de las particularidades.

Aplicamos esta idea a nuestra configuración:

- **Shared expert:** un MLP de dos capas (entrenado con todos los silos).
- **Cabezas especializadas:** una capa lineal por silo.
- **Adaptación:** cuando un silo cambia, **congelamos el shared** y entrenamos solo su cabeza especializada con los nuevos datos (más un buffer de replay con datos antiguos).

## 3. Resultados

Tomamos el silo 0 (variación original 0.05) y generamos un nuevo dominio con variación 0.15 (cambio numérico del 300% en la matriz subyacente). Resultados:

| | Antes | Después | Cambio |
|---|-------|---------|--------|
| Dominio original (0.05) | 0.054668 | 0.039258 | **-28.2%** (mejora) |
| Nuevo dominio (0.15)    | 0.112658 | 0.048091 | **-57.3%** (mejora) |

**No solo no hubo olvido, sino que ambos dominios mejoraron.** El veredicto es:

> ✅ OLVIDO DESPRECIABLE (<5%) + MEJORA EN NUEVOS DATOS

## 4. ¿Por qué funciona?

El shared expert aprende una representación que es **invariante al nivel de variación** (porque fue entrenado con todas las variaciones de 0.05 a 0.50). Al congelarlo, esa representación permanece óptima para cualquier dominio dentro del rango. La cabeza especializada, al ser una simple capa lineal, puede **reproyectar** esas características para adaptarse al nuevo dominio sin dañar el antiguo. De hecho, encuentra una proyección que incluso mejora el dominio original.

## 5. Implicaciones prácticas

| Estrategia | Cuándo usarla |
|------------|----------------|
| **Shared expert + fine-tuning de cabeza** | Cambios graduales dentro del rango visto durante el entrenamiento del shared. |
| **Versionado (crear nuevo experto)** | Cambios drásticos fuera del rango, o cuando se necesita mantener versiones históricas. |

Ambas estrategias pueden coexistir: el versionado se implementa añadiendo un nivel extra al ID jerárquico (ej. `a.b.c.d.e.f.g.h.v2`), mientras que el shared expert se usa para adaptaciones rápidas sin multiplicar parámetros.

## 6. Conclusión

La arquitectura **shared expert + specialized heads** resuelve el problema de olvido catastrófico que enfrentaban los métodos de actualización continua clásicos. Es computacionalmente eficiente (solo se entrena una capa lineal por adaptación) y mantiene la especialización extrema de los expertos identitarios.

Este hallazgo valida la dirección de los modelos híbridos (tronco común + cabezas ligeras) para sistemas MoE en producción.

---

*Para reproducir: `python -m shared_expert.test_shift` en el repositorio [shared-experts-moe](https://github.com/maga-484/shared-experts-moe).*