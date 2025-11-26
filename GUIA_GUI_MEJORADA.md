# 🚀 GUÍA DE USO - GUI MEJORADA CON SELECTOR MT5

## 📁 Archivos Actualizados

1. **improved_strategy_gui.py** - GUI principal con todas las mejoras
2. **find_mt5_installations.py** - Script detector de instalaciones MT5
3. **proven_strategies_library.py** - Documentación de estrategias

## ✨ NUEVAS CARACTERÍSTICAS

### 🔌 Selector de Instalación MT5

**Detección Automática:**
- La GUI detecta automáticamente todas las instalaciones MT5 en tu sistema
- Busca en el Registro de Windows
- Busca en directorios comunes (C:/Program Files, etc.)
- Muestra un dropdown con todas las instalaciones encontradas

**Opciones Disponibles:**
1. **Dropdown**: Selecciona de instalaciones detectadas
2. **Botón "📁 Buscar"**: Navega manualmente si tu instalación no fue detectada
3. **Botón "🔄"**: Actualiza la lista de instalaciones detectadas
4. **Rutas comunes**: Botones de acceso rápido para rutas estándar

### 🔐 Configuración de Credenciales (Opcional)

Si necesitas conectar a una cuenta específica:
- **Login**: Número de cuenta
- **Password**: Contraseña de la cuenta
- **Servidor**: Nombre del servidor (ej: "ICMarkets-Demo")

**Nota**: Dejar vacío para usar la cuenta activa en MT5

### 📊 Dashboard Mejorado

**Información Mostrada:**
- Estado de conexión (con indicador visual)
- Balance de cuenta
- Equity
- Margen usado
- Servidor conectado
- Número de cuenta
- Apalancamiento

**Botones de Control:**
- 🔌 Conectar MT5
- 🔄 Desconectar
- 🔍 Verificar Instalación

## 📝 CÓMO USAR

### Paso 1: Detectar Instalaciones (Opcional)

```bash
python /mnt/user-data/outputs/find_mt5_installations.py
```

Esto te mostrará todas las instalaciones MT5 encontradas y guardará la info en `mt5_installations.json`

### Paso 2: Ejecutar la GUI

```bash
python /mnt/user-data/outputs/improved_strategy_gui.py
```

### Paso 3: Configurar MT5

1. Ve a la pestaña **📊 Dashboard**
2. En **Instalación MT5**:
   - Selecciona una instalación del dropdown (si se detectaron)
   - O haz clic en **📁 Buscar** para navegar manualmente
   - O haz clic en uno de los botones de rutas comunes
3. Si necesitas credenciales específicas, llénalas (opcional)
4. Haz clic en **🔌 Conectar MT5**
5. Verás un mensaje de éxito con información de tu cuenta

### Paso 4: Generar Estrategias

1. Ve a la pestaña **🎯 Autogeneración Mejorada**
2. Configura los parámetros:
   ```
   Símbolos: EURUSD,GBPUSD,USDJPY,AUDUSD
   Timeframe: H4
   Días: 730
   Modo: Estrategias Pre-configuradas (RECOMENDADO)
   Usar ML: ✓
   Refinar: ✗ (para ser más rápido)
   Win Rate mínimo: 42%
   Sharpe mínimo: 0.5
   ```
3. Haz clic en **🚀 Generar Estrategias**
4. Espera 5-10 minutos
5. Revisa los resultados en la tabla

## 🎯 ESTRATEGIAS PRE-CONFIGURADAS

La GUI ahora usa estrategias con parámetros probados:

### Moving Average Crossover:
1. **MA_Classic**: EMA(10) x EMA(20) + RSI(14)
2. **MA_Fast**: EMA(5) x EMA(20) + RSI(14)
3. **MA_Fib**: EMA(8) x EMA(21) + RSI(14)
4. **MA_MACD**: SMA(12) x SMA(26) + RSI(14)

### RSI:
1. **RSI_Classic**: RSI(14) 30/70 + divergencias
2. **RSI_Conservative**: RSI(14) 25/75
3. **RSI_Fast**: RSI(9) 30/70 + divergencias

## 📊 RESULTADOS ESPERADOS

### Con Estrategias Pre-configuradas:
- **28 tests** (4 símbolos × 7 estrategias)
- **Tiempo**: 5-10 minutos
- **Éxito esperado**: 8-14 estrategias viables (30-50%)

### Métricas de Calidad:

**EXCELENTE:**
- Sharpe > 2.0
- Win Rate > 60%
- Profit Factor > 2.0

**BUENO:**
- Sharpe > 1.0
- Win Rate > 55%
- Profit Factor > 1.5

**ACEPTABLE:**
- Sharpe > 0.5
- Win Rate > 50%
- Profit Factor > 1.2

## 🔧 SOLUCIÓN DE PROBLEMAS

### ❌ "No se encontraron instalaciones MT5"

**Solución**:
1. Ejecuta `find_mt5_installations.py` para verificar
2. Si no detecta tu instalación:
   - Usa el botón **📁 Buscar**
   - Navega hasta tu `terminal64.exe`
   - Selecciónalo manualmente

### ❌ "Error conectando a MT5"

**Verifica**:
1. Que la ruta sea correcta (termina en `terminal64.exe`)
2. Que MT5 esté instalado correctamente
3. Que no haya otra aplicación usando MT5
4. Si usas credenciales, que sean correctas

### ❌ "Solo 128 velas" / "Todos Sharpe negativos"

**Solución**:
1. Cambia a **H4** o **H1** (D1 tiene pocas velas disponibles)
2. Aumenta días a **730** (2 años)
3. Usa modo **Estrategias Pre-configuradas**
4. Baja filtros: Win Rate 40%, Sharpe 0.3

### ❌ "0 estrategias viables"

**Posibles causas**:
1. Filtros muy estrictos → Bajar a Win Rate 40%, Sharpe 0.3
2. Pocos datos históricos → Aumentar días
3. Timeframe inadecuado → Probar H4 o H1
4. Mercado difícil → Normal, probar otros símbolos

## 🗂️ INSTALACIONES PORTABLES

Si usas MT5 portable (sin instalar):

1. La GUI puede no detectarlo automáticamente
2. Usa el botón **📁 Buscar**
3. Navega a la carpeta donde descomprimiste MT5
4. Selecciona `terminal64.exe`

Ejemplo de ruta portable:
```
D:/MisAplicaciones/MT5Portable/terminal64.exe
```

## 💾 GUARDAR ESTRATEGIAS VIABLES

1. Después de la generación, haz clic en **💾 Guardar Viables**
2. Las estrategias se guardan en `generated_strategies.json`
3. Incluye:
   - Parámetros de cada estrategia
   - Métricas de performance
   - Símbolo y timeframe

## 📚 PESTAÑAS DE LA GUI

### 🎯 Autogeneración Mejorada
- Configuración de generación
- Tabla de resultados
- Acciones sobre estrategias

### 📚 Biblioteca de Estrategias
- Documentación de estrategias pre-configuradas
- Explicación de parámetros
- Guía de uso

### 📊 Dashboard
- **⚙️ Configuración MT5** (NUEVO)
- **🔐 Credenciales** (NUEVO)
- Conexión/desconexión
- Métricas de cuenta en tiempo real

### 📝 Log
- Registro de todas las operaciones
- Timestamps
- Niveles de severidad (INFO, SUCCESS, ERROR, WARNING)

## 🎓 CONSEJOS

1. **Primera vez**: Ejecuta `find_mt5_installations.py` para ver tus opciones
2. **Múltiples brokers**: Si tienes varias instalaciones MT5, podrás elegir fácilmente
3. **Demo vs Real**: Puedes conectar a diferentes cuentas usando las credenciales
4. **Pruebas iniciales**: Usa H4 con 730 días para tener suficientes datos
5. **Paciencia**: La primera generación puede tomar 10-15 minutos

## 📞 SOPORTE

Si encuentras problemas:
1. Revisa el **Log** (última pestaña)
2. Verifica la ruta MT5 con **🔍 Verificar Instalación**
3. Ejecuta `find_mt5_installations.py` para diagnóstico
4. Intenta conectar manualmente con MT5 para verificar que funciona

## ✅ CHECKLIST DE INICIO

- [ ] MT5 instalado y funcionando
- [ ] Ejecutar `find_mt5_installations.py` (opcional)
- [ ] Ejecutar `improved_strategy_gui.py`
- [ ] Seleccionar instalación MT5 en Dashboard
- [ ] Conectar a MT5 (verificar mensaje de éxito)
- [ ] Configurar parámetros de generación
- [ ] Generar estrategias
- [ ] Revisar resultados
- [ ] Guardar estrategias viables

## 🚀 ¡LISTO PARA EMPEZAR!

Ahora tienes una GUI completa con:
✅ Detección automática de instalaciones MT5
✅ Selector visual de instalaciones
✅ Estrategias pre-configuradas (30-50% éxito)
✅ Dashboard mejorado con todas las métricas
✅ Credenciales opcionales para múltiples cuentas

¡Buena suerte generando tus estrategias! 🎉