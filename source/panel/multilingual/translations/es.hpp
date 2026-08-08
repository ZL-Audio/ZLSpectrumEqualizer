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
    static constexpr std::array kTexts = {
        "Soltar: omitir la banda.",
        "Pulsar: solo de la banda.",
        "Seleccionar el tipo de filtro.",
        "Seleccionar la pendiente del filtro. Una pendiente más alta hace que la curva de respuesta del filtro sea más pronunciada.",
        "Seleccionar el modo estéreo.",
        "Ajustar la frecuencia.",
        "Ajustar la ganancia base y la ganancia objetivo.",
        "Ajustar el factor de calidad. Un valor de Q mayor estrecha el ancho de banda.",
        "Pulsar: activar el comportamiento dinámico espectral.",
        "Clic: desactivar la banda.",

        "Soltar: omitir el procesamiento dinámico espectral.",
        "Pulsar: emitir la señal delta dinámica espectral.",
        "Seleccionar el modo dinámico espectral.",
        "El procesamiento dinámico utiliza un umbral estático.",
        "El procesamiento dinámico utiliza un umbral dinámico relacionado con la sonoridad del sidechain de la banda.",
        "El procesamiento dinámico utiliza un umbral dinámico relacionado con la sonoridad del sidechain total.",
        "Ajustar el umbral estático.",
        "Ajustar el umbral relativo de la banda.",
        "Ajustar el umbral relativo total.",
        "Ajustar el ancho de la rodilla (knee) del procesamiento dinámico.",
        "Ajustar el tiempo de ataque del procesamiento dinámico.",
        "Ajustar el tiempo de liberación del procesamiento dinámico.",

        "Soltar: omitir el plugin.",
        "Pulsar: usar el sidechain externo.\nSoltar: usar el sidechain interno.",

        "Seleccionar la resolución del procesamiento espectral.",
        "Seleccionar el tipo de suavizado del sidechain en el procesamiento espectral.",
        "Seleccionar el valor de suavizado del sidechain en el procesamiento espectral.",
        "Ajustar la pendiente de inclinación (tilt) del sidechain en el procesamiento espectral.",
        "Ajustar el sesgo de ataque del sidechain en el procesamiento espectral. Un sesgo mayor acelera el ataque en altas frecuencias.",
        "Ajustar el sesgo de liberación del sidechain en el procesamiento espectral. Un sesgo mayor acelera la liberación en altas frecuencias.",
        "Ajustar la puerta espectral de sonoridad relativa.",

        "Ajustar la ganancia de salida adicional.",
        "Ajustar la escala de la ganancia base y objetivo de todos los filtros.",
        "Pulsar: activar la Compensación de Ganancia Estática. La SGC es imprecisa, pero no afecta a la dinámica.",
        "Pulsar: comenzar a medir la sonoridad integrada de la señal de entrada y de salida.\nSoltar: actualizar la ganancia de salida con la diferencia entre ambos valores de sonoridad.",

        "Pulsar: activar el analizador de espectro de la señal de entrada.",
        "Pulsar: activar el analizador de espectro de la señal de salida.",
        "Pulsar: activar el analizador de espectro de la señal de sidechain.",
        "Seleccionar la velocidad de caída (decay) de los analizadores de espectro.",
        "Seleccionar la pendiente de inclinación (tilt) de los analizadores de espectro.",
        "Seleccionar el valor de suavizado de los analizadores de espectro.",
        "Seleccionar el tipo de suavizado de los analizadores de espectro.",
        "Pulsar: activar la función de congelación. Pase el ratón sobre el analizador para congelar el espectro.",
        "Pulsar: activar la detección de colisiones.",
        "Ajustar la intensidad de la detección de colisiones.",

        "Doble clic: abrir la configuración de la interfaz.",
    };
}
