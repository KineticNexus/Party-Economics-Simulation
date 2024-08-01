import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class Persona:
    def __init__(self, genero, id):
        self.genero = genero
        self.id = id
        self.atributos = {
            'atractivo': random.randint(1, 10),
            'personalidad': random.randint(1, 10),
            'inteligencia': random.randint(1, 10),
            'gracia': random.randint(1, 10),
            'cultura': random.randint(1, 10),
            'estatus': random.randint(1, 10)
        }
        self.valor_interno = self.calcular_valor_interno()
        self.valor_percibido = self.valor_interno
        self.pareja = None
        self.interacciones = []
        self.nivel_alcohol = 0
        self.umbral_satisfaccion = 1.0
        self.observaciones = {attr: [] for attr in self.atributos}
        self.valoraciones_escasez = {attr: 1.0 for attr in self.atributos}
        self.hora_emparejamiento = None

    def calcular_valor_interno(self):
        pesos = {'atractivo': 0.3, 'personalidad': 0.2, 'inteligencia': 0.15, 
                 'gracia': 0.15, 'cultura': 0.1, 'estatus': 0.1}
        return sum(self.atributos[attr] * pesos[attr] for attr in self.atributos)

    def percibir_valor(self, otra_persona):
        valor_percibido = 0
        for attr, valor in otra_persona.atributos.items():
            desviacion = max(0.5, self.nivel_alcohol * 0.2)
            valor_observado = np.random.normal(valor, desviacion)
            self.observaciones[attr].append(valor_observado)
            self.actualizar_valoracion_escasez(attr)
            valor_percibido += valor_observado * self.valoraciones_escasez[attr]
        return valor_percibido / len(self.atributos)

    def actualizar_valoracion_escasez(self, atributo):
        if len(self.observaciones[atributo]) > 1:
            promedio = np.mean(self.observaciones[atributo])
            ultimo_valor = self.observaciones[atributo][-1]
            if ultimo_valor > promedio:
                escasez = (ultimo_valor - promedio) / promedio
                self.valoraciones_escasez[atributo] = 1 + escasez
            else:
                self.valoraciones_escasez[atributo] = max(0.5, self.valoraciones_escasez[atributo] * 0.95)

    def calcular_ganancia(self, otra_persona):
        valor_percibido = self.percibir_valor(otra_persona)
        return valor_percibido - self.valor_percibido

    def actualizar_valor_percibido(self, nueva_percepcion):
        self.valor_percibido = (self.valor_percibido * len(self.interacciones) + nueva_percepcion) / (len(self.interacciones) + 1)

    def decidir_emparejar(self, otra_persona, tiempo_restante):
        ganancia = self.calcular_ganancia(otra_persona)
        probabilidad = min(1, max(0, (ganancia + self.umbral_satisfaccion) / 2))
        return random.random() < probabilidad

    def beber(self):
        self.nivel_alcohol = min(10, self.nivel_alcohol + random.uniform(0, 1))

    def actualizar_umbral(self, tiempo_restante):
        if tiempo_restante > 0:
            self.umbral_satisfaccion = max(0, self.umbral_satisfaccion - (1 / tiempo_restante))
        else:
            self.umbral_satisfaccion = 0

class Fiesta:
    def __init__(self, num_hombres, num_mujeres, duracion_horas):
        self.hombres = [Persona('H', i) for i in range(num_hombres)]
        self.mujeres = [Persona('M', i) for i in range(num_mujeres)]
        self.duracion = duracion_horas
        self.hora_actual = 0
        self.parejas_por_hora = defaultdict(int)
        self.valor_parejas_por_hora = defaultdict(list)

    def simular(self):
        while self.hora_actual < self.duracion:
            self.interaccion_por_hora()
            self.hora_actual += 1
            for persona in self.hombres + self.mujeres:
                persona.beber()
                persona.actualizar_umbral(self.duracion - self.hora_actual)

    def interaccion_por_hora(self):
        todas_personas = self.hombres + self.mujeres
        for persona in todas_personas:
            if persona.pareja is None:
                posibles_parejas = [p for p in todas_personas if p.genero != persona.genero and p.pareja is None]
                if posibles_parejas:
                    pareja_potencial = self.seleccionar_pareja_potencial(persona, posibles_parejas)
                    if (persona.decidir_emparejar(pareja_potencial, self.duracion - self.hora_actual) and
                        pareja_potencial.decidir_emparejar(persona, self.duracion - self.hora_actual)):
                        persona.pareja = pareja_potencial
                        pareja_potencial.pareja = persona
                        persona.hora_emparejamiento = self.hora_actual
                        pareja_potencial.hora_emparejamiento = self.hora_actual
                        self.parejas_por_hora[self.hora_actual] += 1
                        valor_pareja = persona.valor_interno + pareja_potencial.valor_interno
                        self.valor_parejas_por_hora[self.hora_actual].append(valor_pareja)
                    else:
                        persona.interacciones.append(pareja_potencial)
                        pareja_potencial.interacciones.append(persona)
                        persona.actualizar_valor_percibido(persona.percibir_valor(pareja_potencial))
                        pareja_potencial.actualizar_valor_percibido(pareja_potencial.percibir_valor(persona))

    def seleccionar_pareja_potencial(self, persona, posibles_parejas):
        diferencias = [abs(persona.valor_percibido - p.valor_percibido) for p in posibles_parejas]
        probabilidades = [1 / (1 + d) for d in diferencias]
        probabilidades = [p / sum(probabilidades) for p in probabilidades]
        return np.random.choice(posibles_parejas, p=probabilidades)

    def analizar_resultados(self):
        resultados = {}

        # 1. Valor total de la pareja
        parejas = [(h, h.pareja) for h in self.hombres if h.pareja]
        valores_parejas = [h.valor_interno + m.valor_interno for h, m in parejas]
        resultados['valor_total_parejas'] = np.mean(valores_parejas) if valores_parejas else 0

        # 2 y 3. Cuadros de valoración para hombres y mujeres
        cuadro_hombres = self.crear_cuadro_valoracion(self.hombres)
        cuadro_mujeres = self.crear_cuadro_valoracion(self.mujeres)
        resultados['cuadro_hombres'] = cuadro_hombres['porcentajes']
        resultados['cuadro_mujeres'] = cuadro_mujeres['porcentajes']
        resultados['valores_parejas_hombres'] = cuadro_hombres['valores']
        resultados['valores_parejas_mujeres'] = cuadro_mujeres['valores']
        resultados['promedios_parejas_hombres'] = cuadro_hombres['promedios']
        resultados['promedios_parejas_mujeres'] = cuadro_mujeres['promedios']

        # 4. Tasa de emparejamiento
        emparejados = sum(1 for p in self.hombres + self.mujeres if p.pareja)
        resultados['tasa_emparejamiento'] = emparejados / len(self.hombres + self.mujeres)

        # 5. Diferencia de valor percibido en parejas
        dif_valores = [abs(h.valor_percibido - m.valor_percibido) for h, m in parejas]
        resultados['dif_valor_percibido'] = np.mean(dif_valores) if dif_valores else 0

        # 6. Nivel de alcohol promedio de emparejados vs solteros
        emparejados = [p for p in self.hombres + self.mujeres if p.pareja]
        solteros = [p for p in self.hombres + self.mujeres if not p.pareja]
        resultados['alcohol_emparejados'] = np.mean([p.nivel_alcohol for p in emparejados]) if emparejados else 0
        resultados['alcohol_solteros'] = np.mean([p.nivel_alcohol for p in solteros]) if solteros else 0

        # 7. Número promedio de interacciones antes de emparejarse
        interacciones = [len(p.interacciones) for p in emparejados]
        resultados['interacciones_promedio'] = np.mean(interacciones) if interacciones else 0

        # 8. Distribución de parejas por hora
        resultados['parejas_por_hora'] = dict(self.parejas_por_hora)
        resultados['valor_parejas_por_hora'] = {hora: np.mean(valores) if valores else 0 
                                                for hora, valores in self.valor_parejas_por_hora.items()}

        # 9. Atributo más valorado
        todas_valoraciones = defaultdict(list)
        for p in self.hombres + self.mujeres:
            for attr, valor in p.valoraciones_escasez.items():
                todas_valoraciones[attr].append(valor)
        resultados['atributo_mas_valorado'] = max(todas_valoraciones, key=lambda x: np.mean(todas_valoraciones[x]))

        # 10. Tasa de "conformidad"
        conformes = sum(1 for p in emparejados if p.pareja.valor_percibido < p.valor_interno)
        resultados['tasa_conformidad'] = conformes / len(emparejados) if emparejados else 0

        # Análisis del top 20% más valorado externamente
        todas_personas = self.hombres + self.mujeres
        top_20_percent = sorted(todas_personas, key=lambda p: p.valor_percibido, reverse=True)[:int(len(todas_personas) * 0.2)]
        
        parejas_top_20 = defaultdict(int)
        valores_parejas_top_20 = []
        
        for persona in top_20_percent:
            if persona.pareja:
                categoria_pareja = self.categorizar_valor(persona.pareja.valor_interno, persona.pareja.valor_percibido)
                parejas_top_20[categoria_pareja] += 1
                valores_parejas_top_20.append(persona.valor_interno + persona.pareja.valor_interno)
        
        resultados['parejas_top_20'] = dict(parejas_top_20)
        resultados['valor_promedio_parejas_top_20'] = np.mean(valores_parejas_top_20) if valores_parejas_top_20 else 0

        # Nuevo análisis: valor externo conseguido por cada tipo de personalidad
        valor_externo = {0: [], 1: [], 2: []}  # 0: Subvalorado, 1: Normal, 2: Sobrevalorado
        
        for persona in self.hombres + self.mujeres:
            if persona.pareja:
                categoria = self.categorizar_valor(persona.valor_interno, persona.valor_percibido)
                valor_externo[categoria].append(persona.pareja.valor_interno)

        categorias = ['Subvalorado', 'Normal', 'Sobrevalorado']
        for i, categoria in enumerate(categorias):
            if valor_externo[i]:
                resultados[f'valor_promedio_{categoria}'] = np.mean(valor_externo[i])
            else:
                resultados[f'valor_promedio_{categoria}'] = 0

        return resultados

    def crear_cuadro_valoracion(self, grupo):
        cuadro_porcentajes = np.zeros((3, 3))
        cuadro_valores = np.zeros((3, 3))
        cuadro_conteo = np.zeros((3, 3))
        for persona in grupo:
            if persona.pareja:
                i = self.categorizar_valor(persona.valor_interno, persona.valor_percibido)
                j = self.categorizar_valor(persona.pareja.valor_interno, persona.pareja.valor_percibido)
                cuadro_porcentajes[i][j] += 1
                cuadro_valores[i][j] += persona.valor_interno + persona.pareja.valor_interno
                cuadro_conteo[i][j] += 1
        
        total = len(grupo)
        cuadro_porcentajes = cuadro_porcentajes / total if total > 0 else cuadro_porcentajes
        cuadro_promedios = np.divide(cuadro_valores, cuadro_conteo, where=cuadro_conteo!=0)
        return {'porcentajes': cuadro_porcentajes, 'valores': cuadro_valores, 'promedios': cuadro_promedios}

    @staticmethod
    def categorizar_valor(valor_interno, valor_percibido):
        if valor_interno < valor_percibido:
            return 0  # Subvalorado
        elif valor_interno > valor_percibido:
            return 2  # Sobrevalorado
        else:
            return 1  # Normal

def ejecutar_simulacion():
    fiesta = Fiesta(600, 600, 5)  # 24 horas de duración
    fiesta.simular()
    resultados = fiesta.analizar_resultados()

    # Imprimir resultados numéricos
    print(f"Valor total promedio de las parejas: {resultados['valor_total_parejas']:.2f}")
    print(f"Tasa de emparejamiento: {resultados['tasa_emparejamiento']:.2%}")
    print(f"Diferencia promedio de valor percibido en parejas: {resultados['dif_valor_percibido']:.2f}")
    print(f"Nivel de alcohol promedio de emparejados: {resultados['alcohol_emparejados']:.2f}")
    print(f"Nivel de alcohol promedio de solteros: {resultados['alcohol_solteros']:.2f}")
    print(f"Número promedio de interacciones antes de emparejarse: {resultados['interacciones_promedio']:.2f}")
    print(f"Atributo más valorado: {resultados['atributo_mas_valorado']}")
    print(f"Tasa de conformidad: {resultados['tasa_conformidad']:.2%}")
    print(f"Valor promedio de parejas del top 20%: {resultados['valor_promedio_parejas_top_20']:.2f}")

    # Crear gráficos
    crear_grafico_cuadro_valoracion(resultados['cuadro_hombres'], resultados['valores_parejas_hombres'], resultados['promedios_parejas_hombres'], "Hombres")
    crear_grafico_cuadro_valoracion(resultados['cuadro_mujeres'], resultados['valores_parejas_mujeres'], resultados['promedios_parejas_mujeres'], "Mujeres")
    crear_grafico_parejas_por_hora(resultados['parejas_por_hora'], resultados['valor_parejas_por_hora'])
    crear_grafico_parejas_top_20(resultados['parejas_top_20'])
    crear_grafico_valor_externo_por_tipo(resultados)

def crear_grafico_cuadro_valoracion(cuadro_porcentajes, cuadro_valores, cuadro_promedios, genero):
    categorias = ['Subvalorado', 'Normal', 'Sobrevalorado']
    plt.figure(figsize=(14, 12))
    sns.heatmap(cuadro_porcentajes, annot=True, fmt='.2%', cmap='YlGnBu', xticklabels=categorias, yticklabels=categorias)
    
    for i in range(3):
        for j in range(3):
            plt.text(j + 0.5, i + 0.75, f'Total: {cuadro_valores[i, j]:.2f}', 
                     ha='center', va='center', color='red', fontweight='bold')
            plt.text(j + 0.5, i + 0.25, f'Prom: {cuadro_promedios[i, j]:.2f}', 
                     ha='center', va='center', color='green', fontweight='bold')
    
    plt.title(f"Cuadro de valoración para {genero}")
    plt.xlabel("Categoría de la pareja")
    plt.ylabel("Categoría propia")
    plt.show()

def crear_grafico_parejas_por_hora(parejas_por_hora, valor_parejas_por_hora):
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.bar(parejas_por_hora.keys(), parejas_por_hora.values(), color='b', alpha=0.5)
    ax1.set_xlabel('Hora')
    ax1.set_ylabel('Número de parejas', color='b')
    ax1.tick_params(axis='y', labelcolor='b')

    ax2 = ax1.twinx()
    horas = sorted(valor_parejas_por_hora.keys())
    valores = [valor_parejas_por_hora[h] for h in horas]
    ax2.plot(horas, valores, color='r', marker='o')
    ax2.set_ylabel('Valor promedio de las parejas', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.title("Distribución de parejas y su valor promedio por hora")
    plt.show()

def crear_grafico_parejas_top_20(parejas_top_20):
    categorias = ['Subvalorado', 'Normal', 'Sobrevalorado']
    valores = [parejas_top_20.get(i, 0) for i in range(3)]
    plt.figure(figsize=(10, 6))
    plt.bar(categorias, valores)
    plt.title("Distribución de parejas del top 20% más valorado externamente")
    plt.xlabel("Categoría de la pareja")
    plt.ylabel("Número de parejas")
    for i, v in enumerate(valores):
        plt.text(i, v, str(v), ha='center', va='bottom')
    plt.show()

def crear_grafico_valor_externo_por_tipo(resultados):
    categorias = ['Subvalorado', 'Normal', 'Sobrevalorado']
    valores = [resultados[f'valor_promedio_{cat}'] for cat in categorias]

    plt.figure(figsize=(10, 6))
    plt.bar(categorias, valores)
    plt.title("Valor Externo Promedio Conseguido por Tipo de Personalidad")
    plt.xlabel("Tipo de Personalidad")
    plt.ylabel("Valor Externo Promedio")
    for i, v in enumerate(valores):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    plt.show()

if __name__ == "__main__":
    ejecutar_simulacion()