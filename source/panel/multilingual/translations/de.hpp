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

namespace zlpanel::multilingual::de {
    inline constexpr std::array kTexts = {
        // band basic controls
        "Aktiviert/deaktiviert den Band-Bypass.",
        "Aktiviert/deaktiviert das Solo-Abhören für dieses Band.",
        "Wählt den Filtertyp aus.",
        "Wählt die Filterflanke aus. Eine höhere Flankensteilheit erzeugt eine steilere Frequenzgangkurve.",
        "Wählt den Stereo-Kanalmodus aus.",
        "Regelt die Mitten- oder Grenzfrequenz.",
        "Regelt die Grundverstärkung und die dynamische Zielverstärkung.",
        "Regelt den Gütefaktor (Q). Ein höherer Q-Wert verringert die Bandbreite.",
        "Aktiviert/deaktiviert das dynamische EQ-Verhalten für dieses Band.",
        "Schaltet dieses Band aus.",
        "Wählt das vorherige Band aus.",
        "Wählt das nächste Band aus.",
        "Aktiviert/deaktiviert die Pass-Skalierung für Tiefpass- und Hochpassfilter.",
        "Regelt die Amplitudenskalierung des Passfilters.",
        // band dynamic controls
        "Aktiviert/deaktiviert den Bypass der Dynamikbearbeitung für dieses Band.",
        "Schaltet die Ausgabe des Spektraldynamik-Delta-Signals um.",
        "Wählt den Spektraldynamikmodus aus.",
        "Verwendet einen statischen Schwellenwert für die Dynamikbearbeitung.",
        "Verwendet einen dynamischen Schwellenwert basierend auf der Band-Sidechain-Lautheit.",
        "Verwendet einen dynamischen Schwellenwert basierend auf der Gesamt-Sidechain-Lautheit.",
        "Regelt den statischen Schwellenwert.",
        "Regelt den relativen Band-Schwellenwert.",
        "Regelt den relativen Gesamtschwellenwert.",
        "Regelt die Kniebreite (Knee) der Dynamikbearbeitung.",
        "Regelt die Attack-Zeit der Dynamikbearbeitung.",
        "Regelt die Release-Zeit der Dynamikbearbeitung.",
        // top controls
        "Blendet das Bedienfeld für UI-Einstellungen ein/aus.",
        "Blendet den Preset-Browser ein/aus.",
        "Blendet das Bedienfeld für Analyzer-Einstellungen ein/aus.",
        "Blendet das Bedienfeld für Ausgangseinstellungen ein/aus.",
        "Aktiviert/deaktiviert den Plugin-Bypass.",
        "Aktiviert/deaktiviert den externen Side-Chain-Eingang.",
        // preset browser
        "Öffnet den Preset-Ordner im Dateimanager des Systems.",
        "Löscht die ausgewählte Preset-Gruppe.",
        "Löscht das ausgewählte Preset.",
        "Schließt den Preset-Browser.",
        // spectrum controls
        "Blendet die Einstellungen zur Spektralverarbeitung ein oder aus.",
        "Wählt die Auflösung der Spektralverarbeitung aus.",
        "Wählt den Modus der Frequenzglättung der Sidechain für die Spektralverarbeitung aus.",
        "Regelt die Stärke der Frequenzglättung der Sidechain für die Spektralverarbeitung.",
        "Regelt die Tilt-Neigung der Sidechain für die Spektralverarbeitung.",
        "Regelt den Attack-Skew der Sidechain für die Spektralverarbeitung. Ein höherer Skew beschleunigt den Attack bei hohen Frequenzen.",
        "Regelt den Release-Skew der Sidechain für die Spektralverarbeitung. Ein höherer Skew beschleunigt den Release bei hohen Frequenzen.",
        "Regelt das Gate der relativen Lautheit für die Spektralverarbeitung.",
        // output controls
        "Regelt die zusätzliche Ausgangsverstärkung.",
        "Regelt die Gesamtskalierung der Grund- und Zielverstärkung aller Filter.",
        "Aktiviert/deaktiviert die statische Gain-Kompensation (SGC). SGC ist näherungsweise, beeinflusst aber nicht die Dynamik.",
        "Aktiviert/deaktiviert die Lautheitsanpassung. Beim Aktivieren wird die Ein- und Ausgangslautheit gemessen; beim Deaktivieren wird die Ausgangsverstärkung auf deren Differenz gesetzt.",
        // analyzer controls
        "Blendet das Eingangssignal-Spektrum ein/aus.",
        "Blendet das Ausgangssignal-Spektrum ein/aus.",
        "Blendet das Side-Chain-Signal-Spektrum ein/aus.",
        "Wählt die Abklinggeschwindigkeit des Spektrumanalysators aus.",
        "Wählt die Neigung (Tilt) des Spektrumanalysators aus.",
        "Wählt die Frequenzglättungsbreite des Spektrumanalysators aus.",
        "Wählt den Modus der Frequenzglättung aus.\nOCT: konstante Oktavbruchteil-Glättung.\nERB: Glättung nach äquivalenter Rechteckbandbreite.",
        "Aktiviert/deaktiviert das Einfrieren des Spektrums beim Stillhalten des Mauszeigers über dem Analyzer. Wenn aktiviert, kann das Einfrieren auch über Shift umgeschaltet werden.",
        "Wählt den vom Spektrumanalysator angezeigten Stereo-Kanalmodus aus.",
        "Aktiviert/deaktiviert die Erkennung von Frequenzkollisionen.",
        "Regelt die Stärke der Kollisionserkennung.",
        "Blendet die numerische Wertanzeige ein/aus.",
        "Blendet das Pegelmesser-Bedienfeld ein/aus.",
        // curve scale & piano roll
        "Wählt den Dezibel-Anzeigebereich für die EQ-Frequenzgangkurve aus.",
        "Wählt die obere Dezibel-Grenze für die Analyzer-Anzeige aus.",
        "Wählt die minimale Dezibel-Untergrenze für die Analyzer-Anzeige aus.",
        "Blendet die Piano-Roll ein/aus.",
        " "
    };
}
