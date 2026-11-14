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

namespace zlpanel::multilingual::ja {
    inline constexpr std::array kTexts = {
        // band basic controls
        "バンドのバイパス状態を切り替えます。",
        "バンドのソロ試聴を切り替えます。",
        "フィルタータイプを選択します。",
        "フィルタースロープを選択します。スロープが高いほど、周波数レスポンスカーブの変化が急になります。",
        "ステレオチャンネルモードを選択します。",
        "中心周波数またはカットオフ周波数を調整します。",
        "ベースゲインとダイナミックターゲットゲインを調整します。",
        "Qファクターを調整します。Q値が大きいほど帯域幅が狭くなります。",
        "このバンドのダイナミックEQ動作を切り替えます。",
        "このバンドをオフにします。",
        "前のバンドを選択します。",
        "次のバンドを選択します。",
        "ローパスおよびハイパスフィルターのパススケーリングを切り替えます。",
        "パスフィルターの振幅スケールを調整します。",
        // band dynamic controls
        "このバンドのダイナミック処理バイパスを切り替えます。",
        "スペクトラム・ダイナミック・デルタ信号の出力を切り替えます。",
        "スペクトラム・ダイナミックモードを選択します。",
        "ダイナミック処理でスタティック・スレッショルドを使用します。",
        "ダイナミック処理でバンドのサイドチェーン・ラウドネスに応じた動的スレッショルドを使用します。",
        "ダイナミック処理でトータル・サイドチェーンのラウドネスに応じた動的スレッショルドを使用します。",
        "スタティック・スレッショルドを調整します。",
        "バンドの相対スレッショルドを調整します。",
        "トータルの相対スレッショルドを調整します。",
        "ダイナミック処理のニー幅を調整します。",
        "ダイナミック処理のアタックタイムを調整します。",
        "ダイナミック処理のリリースタイムを調整します。",
        // top controls
        "UI設定パネルの表示を切り替えます。",
        "プリセットブラウザの表示を切り替えます。",
        "アナライザー設定パネルの表示を切り替えます。",
        "出力設定パネルの表示を切り替えます。",
        "プラグインのバイパス状態を切り替えます。",
        "外部サイドチェーン入力を切り替えます。",
        // preset browser
        "OSのファイルマネージャーでプリセットフォルダーを開きます。",
        "選択したプリセットグループを削除します。",
        "選択したプリセットを削除します。",
        "プリセットブラウザを閉じます。",
        // spectrum controls
        "スペクトラム処理の設定パネルを表示／非表示にします。",
        "スペクトラム処理の解像度を選択します。",
        "スペクトラム処理のサイドチェーンの周波数スムージングモードを選択します。",
        "スペクトラム処理のサイドチェーンの周波数スムージング量を調整します。",
        "スペクトラム処理のサイドチェーン・ティルトスロープを調整します。",
        "スペクトラム処理のサイドチェーン・アタック・スキューを調整します。スキュー値が高いほど、高周波数帯のアタックが速くなります。",
        "スペクトラム処理のサイドチェーン・リリース・スキューを調整します。スキュー値が高いほど、高周波数帯のリリースが速くなります。",
        "スペクトラム処理の相対ラウドネス・ゲートを調整します。",
        // output controls
        "追加の出力ゲインを調整します。",
        "全フィルターのベースゲインとターゲットゲインの全体スケールを調整します。",
        "静的ゲイン補正（SGC）を切り替えます。SGCは概算ですが、ダイナミクスには影響しません。",
        "ラウドネスゲインマッチングを切り替えます。オンにすると入出力のラウドネスを測定し、オフにするとその差分を出力ゲインに反映します。",
        // analyzer controls
        "入力信号スペクトラムの表示を切り替えます。",
        "出力信号スペクトラムの表示を切り替えます。",
        "サイドチェーン信号スペクトラムの表示を切り替えます。",
        "スペクトラムアナライザーの減衰速度を選択します。",
        "スペクトラムアナライザーのティルトスロープを選択します。",
        "スペクトラムアナライザーの周波数スムージング幅を選択します。",
        "スペクトラムアナライザーの周波数スムージングモードを選択します。\nOCT: 等比分数オクターブによるスムージング。\nERB: 聴覚特性に基づく等価矩形帯域幅スムージング。",
        "アナライザー上にポインターを合わせた際のスペクトラム凍結を切り替えます。",
        "スペクトラムアナライザーが表示するステレオチャンネルモードを選択します。",
        "周波数衝突検出を切り替えます。",
        "衝突検出の強度を調整します。",
        "数値表示の表示を切り替えます。",
        "レベルメーターパネルの表示を切り替えます。",
        // curve scale & piano roll
        "EQレスポンスカーブのデシベル表示範囲を選択します。",
        "スペクトラムアナライザー表示の上限デシベルを選択します。",
        "スペクトラムアナライザー表示の下限デシベルを選択します。",
        "ピアノロールの表示を切り替えます。",
        " "
    };
}
