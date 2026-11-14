// Copyright (C) 2026 - zsliu98
// This file is part of ZLSpectrumEqualizer
//
// ZLSpectrumEqualizer is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License Version 3 as published by the Free Software Foundation.
//
// ZLSpectrumEqualizer is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License along with ZLSpectrumEqualizer. If not, see <https://www.gnu.org/licenses/>.

// This file is also dual licensed under the Apache License, Version 2.0. You may obtain a copy of the License at <http://www.apache.org/licenses/LICENSE-2.0>

#pragma once

#include <array>

namespace zlpanel::multilingual::es {
    inline constexpr std::array kTexts = {
        // band basic controls
        "Activa/desactiva el bypass de la banda.",
        "Activa/desactiva la escucha en solo de la banda.",
        "Selecciona el tipo de filtro.",
        "Selecciona la pendiente del filtro. Una pendiente mayor produce una curva de respuesta en frecuencia más abrupta.",
        "Selecciona el modo de canal estéreo.",
        "Ajusta la frecuencia central o de corte.",
        "Ajusta la ganancia base y la ganancia objetivo dinámica.",
        "Ajusta el factor Q. Un valor Q mayor estrecha el ancho de banda.",
        "Activa/desactiva el comportamiento dinámico de ecualización para esta banda.",
        "Desactiva esta banda.",
        "Selecciona la banda anterior.",
        "Selecciona la banda siguiente.",
        "Activa/desactiva el escalado de paso para los filtros paso bajo y paso alto.",
        "Ajusta la escala de magnitud del filtro de paso.",
        // band dynamic controls
        "Activa/desactiva el bypass del procesamiento dinámico para esta banda.",
        "Activa o desactiva la salida de la señal delta de la dinámica espectral.",
        "Selecciona el modo dinámico espectral.",
        "Utiliza un umbral estático para el procesamiento dinámico.",
        "Utiliza un umbral dinámico basado en la sonoridad del sidechain de la banda.",
        "Utiliza un umbral dinámico basado en la sonoridad del sidechain total.",
        "Ajusta el umbral estático.",
        "Ajusta el umbral relativo de la banda.",
        "Ajusta el umbral relativo total.",
        "Ajusta la anchura del codo (Knee) del procesamiento dinámico.",
        "Ajusta el tiempo de ataque del procesamiento dinámico.",
        "Ajusta el tiempo de relajación del procesamiento dinámico.",
        // top controls
        "Muestra/oculta el panel de ajustes de la interfaz.",
        "Muestra/oculta el explorador de preajustes.",
        "Muestra/oculta el panel de ajustes del analizador.",
        "Muestra/oculta el panel de ajustes de salida.",
        "Activa/desactiva el bypass del plugin.",
        "Activa/desactiva la entrada de side-chain externo.",
        // preset browser
        "Abre la carpeta de preajustes en el administrador de archivos del sistema.",
        "Elimina el grupo de preajustes seleccionado.",
        "Elimina el preajuste seleccionado.",
        "Cierra el explorador de preajustes.",
        // spectrum controls
        "Muestra u oculta los ajustes del procesamiento espectral.",
        "Selecciona la resolución del procesamiento espectral.",
        "Selecciona el modo de suavizado en frecuencia del sidechain para el procesamiento espectral.",
        "Ajusta la cantidad de suavizado en frecuencia del sidechain para el procesamiento espectral.",
        "Ajusta la pendiente de inclinación (tilt) del sidechain para el procesamiento espectral.",
        "Ajusta el sesgo de ataque del sidechain para el procesamiento espectral. Un sesgo mayor acelera el ataque en altas frecuencias.",
        "Ajusta el sesgo de liberación del sidechain para el procesamiento espectral. Un sesgo mayor acelera la liberación en altas frecuencias.",
        "Ajusta la puerta de sonoridad relativa para el procesamiento espectral.",
        // output controls
        "Ajusta la ganancia de salida adicional.",
        "Ajusta la escala general de la ganancia base y objetivo de todos los filtros.",
        "Activa/desactiva la Compensación de Ganancia Estática (SGC). SGC es aproximada, pero no afecta a la dinámica.",
        "Activa/desactiva la igualación de ganancia por sonoridad. Al activarla mide la sonoridad de entrada y salida; al desactivarla ajusta la ganancia de salida según su diferencia.",
        // analyzer controls
        "Muestra/oculta el espectro de la señal de entrada.",
        "Muestra/oculta el espectro de la señal de salida.",
        "Muestra/oculta el espectro de la señal de side-chain.",
        "Selecciona la velocidad de decaimiento del analizador de espectro.",
        "Selecciona la pendiente de inclinación (tilt) del analizador de espectro.",
        "Selecciona el ancho de suavizado frecuencial del analizador de espectro.",
        "Selecciona el modo de suavizado frecuencial.\nOCT: suavizado por fracción de octava constante.\nERB: suavizado por ancho de banda rectangular equivalente.",
        "Activa/desactiva la congelación del espectro al pasar el puntero sobre el analizador.",
        "Selecciona el modo de canal estéreo mostrado por el analizador de espectro.",
        "Activa/desactiva la detección de colisiones de frecuencia.",
        "Ajusta la intensidad de la detección de colisiones.",
        "Muestra/oculta la lectura numérica de valores.",
        "Muestra/oculta el panel de medidores de nivel.",
        // curve scale & piano roll
        "Selecciona el rango de visualización en decibelios para la curva de respuesta del ecualizador.",
        "Selecciona el límite superior en decibelios para la visualización del analizador de espectro.",
        "Selecciona el nivel mínimo en decibelios para la visualización del analizador de espectro.",
        "Muestra/oculta el piano roll.",
        " "
    };
}
