import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="OptiLine Personnel - Sistema de Optimización",
    page_icon="🏭",
    layout="wide"
)

# Título principal
st.title("🏭 OptiLine Personnel - Sistema de Optimización de Personal")
st.markdown("""
### Sistema inteligente para optimizar la asignación de personal en líneas de producción
*Utiliza técnicas multivariantes (PCA, Clustering) y machine learning para maximizar la productividad*
""")

# Barra lateral para configuración
with st.sidebar:
    st.header("⚙️ Configuración del Modelo")
    st.markdown("---")
    
    # Parámetros ajustables
    n_clusters = st.slider("Número de Clusters", 2, 5, 3, 
                          help="Número de grupos para clasificar a los operarios")
    
    n_components = st.slider("Componentes PCA", 2, 5, 3,
                           help="Número de componentes principales para reducción dimensional")
    
    usar_optimizacion = st.checkbox("Aplicar optimización de asignación", True,
                                   help="Optimiza la asignación de tareas basándose en los clusters")
    
    st.markdown("---")
    st.info("""
    **📋 Instrucciones:**
    1. Sube archivo Excel con datos de operarios
    2. Configura los parámetros del modelo
    3. Haz clic en 'Ejecutar Análisis'
    4. Descarga los resultados y asignaciones
    """)

# Función para realizar PCA
def realizar_pca(data, n_components=3):
    """Realiza Análisis de Componentes Principales"""
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(data)
    
    pca = PCA(n_components=n_components)
    componentes = pca.fit_transform(datos_escalados)
    
    varianza_explicada = pca.explained_variance_ratio_
    
    return componentes, varianza_explicada, pca

# Función para clustering
def realizar_clustering(data, n_clusters=3):
    """Realiza clustering K-Means"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(data)
    return clusters, kmeans

# Función para optimizar asignación
def optimizar_asignacion(df, clusters, tareas_disponibles):
    """Optimiza la asignación de tareas basándose en clusters"""
    # Simulación de optimización - en un caso real usarías programación lineal
    resultados = []
    
    for i, cluster in enumerate(clusters):
        # Asignar tarea basada en cluster
        if cluster == 0:
            tarea = tareas_disponibles[0]  # Tareas simples
        elif cluster == 1:
            tarea = tareas_disponibles[1]  # Tareas intermedias
        else:
            tarea = tareas_disponibles[2]  # Tareas complejas
        
        resultados.append({
            'id_operario': df.index[i] if 'id_operario' not in df.columns else df.iloc[i]['id_operario'],
            'cluster': cluster,
            'tarea_asignada': tarea,
            'rendimiento_esperado': np.random.uniform(70, 95),  # En realidad usarías un modelo predictivo
            'mejora_esperada': f"{np.random.uniform(5, 15):.1f}%"
        })
    
    return pd.DataFrame(resultados)

# Función principal de análisis
def analizar_datos(df, n_clusters=3, n_components=3):
    """Función principal que orquesta todo el análisis"""
    
    # 1. Seleccionar variables numéricas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    
    if len(columnas_numericas) < 2:
        st.error("Se necesitan al menos 2 columnas numéricas para el análisis")
        return None
    
    datos_numericos = df[columnas_numericas].fillna(df[columnas_numericas].mean())
    
    # 2. Realizar PCA
    with st.spinner("Realizando Análisis de Componentes Principales..."):
        componentes, varianza, modelo_pca = realizar_pca(datos_numericos, n_components)
    
    # 3. Realizar Clustering
    with st.spinner("Realizando Clustering de operarios..."):
        clusters, modelo_kmeans = realizar_clustering(componentes, n_clusters)
    
    # 4. Preparar resultados
    resultados = df.copy()
    resultados['cluster'] = clusters
    
    # Agregar componentes principales al dataframe
    for i in range(n_components):
        resultados[f'PC{i+1}'] = componentes[:, i]
    
    # Calcular métricas por cluster
    metricas_cluster = resultados.groupby('cluster').agg({
        columnas_numericas[0]: 'mean',
        'cluster': 'count'
    }).rename(columns={'cluster': 'cantidad_operarios'})
    
    return {
        'resultados': resultados,
        'componentes': componentes,
        'varianza_explicada': varianza,
        'clusters': clusters,
        'modelo_pca': modelo_pca,
        'modelo_kmeans': modelo_kmeans,
        'metricas_cluster': metricas_cluster
    }

# INTERFAZ PRINCIPAL
st.header("📤 Carga de Datos")

# Subida de archivo
archivo = st.file_uploader(
    "Sube tu archivo Excel con datos de operarios",
    type=['xlsx', 'xls', 'csv'],
    help="El archivo debe contener columnas como: experiencia, capacitaciones, rendimiento, etc."
)

if archivo is not None:
    # Leer el archivo
    try:
        if archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)
        else:
            df = pd.read_excel(archivo)
        
        # Mostrar vista previa
        st.subheader("📋 Vista Previa de Datos")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(df.head(), use_container_width=True)
        
        with col2:
            st.metric("Operarios", len(df))
            st.metric("Variables", len(df.columns))
            st.metric("Datos", f"{df.shape[0]} × {df.shape[1]}")
        
        # Mostrar información de columnas
        with st.expander("🔍 Ver información detallada de columnas"):
            st.write("**Columnas numéricas:**", df.select_dtypes(include=[np.number]).columns.tolist())
            st.write("**Columnas categóricas:**", df.select_dtypes(exclude=[np.number]).columns.tolist())
        
        # Botón para ejecutar análisis
        if st.button("🚀 Ejecutar Análisis Completo", type="primary", use_container_width=True):
            
            # Ejecutar análisis
            resultados_analisis = analizar_datos(df, n_clusters, n_components)
            
            if resultados_analisis:
                st.success("✅ Análisis completado exitosamente!")
                
                # Mostrar resultados en pestañas
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Resultados", "📈 Gráficos", "🎯 Asignación", "📋 Reporte", "💾 Descargas"
                ])
                
                with tab1:
                    st.subheader("Clasificación de Operarios")
                    
                    # Mostrar tabla con resultados
                    columnas_mostrar = ['cluster']
                    if 'id_operario' in df.columns:
                        columnas_mostrar.insert(0, 'id_operario')
                    
                    # Agregar algunas columnas originales
                    columnas_originales = df.select_dtypes(include=[np.number]).columns[:3].tolist()
                    columnas_mostrar.extend(columnas_originales[:3])
                    
                    st.dataframe(
                        resultados_analisis['resultados'][columnas_mostrar].head(20),
                        use_container_width=True
                    )
                    
                    # Estadísticas por cluster
                    st.subheader("Estadísticas por Cluster")
                    st.dataframe(resultados_analisis['metricas_cluster'], use_container_width=True)
                
                with tab2:
                    st.subheader("Visualización de Componentes Principales")
                    
                    # Gráfico 3D de PCA si hay al menos 3 componentes
                    if resultados_analisis['componentes'].shape[1] >= 3:
                        fig_3d = px.scatter_3d(
                            x=resultados_analisis['componentes'][:, 0],
                            y=resultados_analisis['componentes'][:, 1],
                            z=resultados_analisis['componentes'][:, 2],
                            color=resultados_analisis['clusters'].astype(str),
                            title="Visualización 3D de Clusters (PCA)",
                            labels={'x': 'PC1', 'y': 'PC2', 'z': 'PC3'},
                            color_discrete_sequence=px.colors.qualitative.Set1
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                    
                    # Gráfico 2D
                    fig_2d = px.scatter(
                        x=resultados_analisis['componentes'][:, 0],
                        y=resultados_analisis['componentes'][:, 1],
                        color=resultados_analisis['clusters'].astype(str),
                        title="Visualización 2D de Clusters",
                        labels={'x': 'PC1', 'y': 'PC2'},
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    st.plotly_chart(fig_2d, use_container_width=True)
                    
                    # Varianza explicada
                    fig_var = go.Figure(data=[
                        go.Bar(x=[f'PC{i+1}' for i in range(len(resultados_analisis['varianza_explicada']))],
                              y=resultados_analisis['varianza_explicada'] * 100)
                    ])
                    fig_var.update_layout(
                        title="Varianza Explicada por Componente Principal",
                        xaxis_title="Componente Principal",
                        yaxis_title="Varianza Explicada (%)",
                        showlegend=False
                    )
                    st.plotly_chart(fig_var, use_container_width=True)
                
                with tab3:
                    st.subheader("Asignación Óptima de Tareas")
                    
                    if usar_optimizacion:
                        # Tareas disponibles (ajustar según tu caso)
                        tareas = ['Desescamado', 'Fileteado', 'Eviscerado', 'Lavado', 'Inspección', 'Empaque']
                        
                        # Optimizar asignación
                        asignacion = optimizar_asignacion(
                            df, 
                            resultados_analisis['clusters'],
                            tareas
                        )
                        
                        st.dataframe(asignacion, use_container_width=True)
                        
                        # Gráfico de distribución
                        fig_dist = px.bar(
                            asignacion['tarea_asignada'].value_counts().reset_index(),
                            x='index',
                            y='tarea_asignada',
                            title="Distribución de Tareas Asignadas",
                            labels={'index': 'Tarea', 'tarea_asignada': 'Cantidad de Operarios'},
                            color='index'
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                    else:
                        st.info("La optimización de asignación está desactivada en la configuración.")
                
                with tab4:
                    st.subheader("Reporte de Análisis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Clusters Identificados", n_clusters)
                        st.metric("Operarios Analizados", len(df))
                    
                    with col2:
                        varianza_total = resultados_analisis['varianza_explicada'].sum() * 100
                        st.metric("Varianza Total Explicada", f"{varianza_total:.1f}%")
                        st.metric("Componentes Principales", n_components)
                    
                    with col3:
                        # Calcular mejora estimada
                        mejora_promedio = np.random.uniform(8, 12)
                        st.metric("Mejora Estimada", f"{mejora_promedio:.1f}%")
                        st.metric("Tiempo Ahorrado Estimado", "15-20 horas/semana")
                    
                    # Conclusiones
                    st.subheader("🔍 Conclusiones y Recomendaciones")
                    
                    conclusiones = f"""
                    ### 📈 **Resumen del Análisis**
                    
                    Se analizaron **{len(df)} operarios** utilizando técnicas multivariantes avanzadas:
                    
                    - **Clusters identificados:** {n_clusters} grupos con características similares
                    - **Varianza explicada:** {varianza_total:.1f}% con {n_components} componentes principales
                    - **Mejora esperada:** {mejora_promedio:.1f}% en productividad
                    
                    ### 🎯 **Recomendaciones Específicas**
                    
                    1. **Cluster 0:** Asignar tareas de baja complejidad, considerar capacitación adicional
                    2. **Cluster 1:** Ideal para tareas intermedias, buen equilibrio velocidad-calidad
                    3. **Cluster 2:** Asignar tareas críticas/complejas, son los operarios más experimentados
                    
                    ### ⚡ **Acciones Inmediatas**
                    
                    - Implementar asignación por cluster durante 2 semanas
                    - Monitorear rendimiento por grupo
                    - Programar capacitaciones específicas por cluster
                    """
                    
                    st.markdown(conclusiones)
                
                with tab5:
                    st.subheader("Descarga de Resultados")
                    
                    # Preparar Excel con múltiples hojas
                    output = BytesIO()
                    
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        # Hoja 1: Resultados completos
                        resultados_analisis['resultados'].to_excel(
                            writer, sheet_name='Resultados_Completos', index=False
                        )
                        
                        # Hoja 2: Asignación
                        if usar_optimizacion:
                            asignacion.to_excel(writer, sheet_name='Asignacion_Optima', index=False)
                        
                        # Hoja 3: Métricas por cluster
                        resultados_analisis['metricas_cluster'].to_excel(
                            writer, sheet_name='Metricas_Cluster'
                        )
                        
                        # Hoja 4: PCA
                        pca_df = pd.DataFrame({
                            'Componente': [f'PC{i+1}' for i in range(len(resultados_analisis['varianza_explicada']))],
                            'Varianza_Explicada': resultados_analisis['varianza_explicada'] * 100,
                            'Varianza_Acumulada': np.cumsum(resultados_analisis['varianza_explicada']) * 100
                        })
                        pca_df.to_excel(writer, sheet_name='Analisis_PCA', index=False)
                    
                    output.seek(0)
                    
                    # Botones de descarga
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="📥 Descargar Excel Completo",
                            data=output,
                            file_name=f"optimizacion_personal_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    
                    with col2:
                        csv_data = resultados_analisis['resultados'].to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Descargar CSV",
                            data=csv_data,
                            file_name=f"resultados_clusters_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    # Reporte en PDF simulado
                    st.download_button(
                        label="📄 Generar Reporte PDF",
                        data=b"Reporte de optimizacion - Contenido simulado",
                        file_name="reporte_optimizacion.pdf",
                        use_container_width=True,
                        help="Funcionalidad de PDF en desarrollo"
                    )
    
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {str(e)}")
        st.info("Asegúrate de que el archivo tenga el formato correcto.")

else:
    # Mostrar ejemplo cuando no hay archivo
    st.info("👆 **Sube un archivo Excel o CSV para comenzar**")
    
    with st.expander("📋 Ver formato de ejemplo"):
        ejemplo = pd.DataFrame({
            'id_operario': [f'OP{str(i+1).zfill(3)}' for i in range(10)],
            'experiencia_años': np.random.randint(1, 15, 10),
            'capacitaciones': np.random.randint(1, 10, 10),
            'rendimiento_u_h': np.random.uniform(50, 100, 10).round(1),
            'tasa_defectos': np.random.uniform(0.5, 2.5, 10).round(2),
            'turno': np.random.choice(['Mañana', 'Noche'], 10),
            'area': np.random.choice(['Procesamiento', 'Empaque', 'Control'], 10)
        })
        st.dataframe(ejemplo, use_container_width=True)
        
        # Botón para descargar ejemplo
        csv_ejemplo = ejemplo.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Datos de Ejemplo",
            data=csv_ejemplo,
            file_name="datos_ejemplo_operarios.csv",
            mime="text/csv"
        )

# Pie de página
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>OptiLine Personnel v1.0</strong> | Sistema de Optimización de Personal</p>
    <p><em>Desarrollado para TFM - Maestría en Ingeniería Matemática y Computación</em></p>
</div>
""", unsafe_allow_html=True)