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
    static constexpr std::array kTexts = {
        "離す: バンドをバイパス。",
        "押す: バンドをソロ再生。",
        "フィルタータイプを選択。",
        "フィルタースロープを選択。スロープが高くなると、フィルターの応答曲線がより急になります。",
        "ステレオモードを選択。",
        "周波数を調整。",
        "ベースゲインとターゲットゲインを調整。",
        "Q値を調整。Q値が大きいほど帯域幅が狭くなります。",
        "押す: ダイナミック動作をオン。",
        "クリック: バンドをオフ。",

        "離す: ダイナミック処理をバイパス。",
        "ダイナミックモードを選択。",
        "ダイナミック処理でスタティック・スレッショルドを使用。",
        "ダイナミック処理でバンドのサイドチェーン・ラウドネスに応じた動的スレッショルドを使用。",
        "ダイナミック処理でトータル・サイドチェーンのラウドネスに応じた動的スレッショルドを使用。",
        "スタティック・スレッショルドを調整。",
        "バンドの相対スレッショルドを調整。",
        "トータルの相対スレッショルドを調整。",
        "ダイナミック処理のニー幅を調整。",
        "ダイナミック処理のアタックタイムを調整。",
        "ダイナミック処理のリリースタイムを調整。",

        "離す: プラグインをバイパス。",
        "押す: 外部サイドチェーンを使用。\n離す: 内部サイドチェーンを使用。",

        "スペクトラム処理の解像度を選択。",
        "スペクトラム処理のサイドチェーン・スムージングタイプを選択。",
        "スペクトラム処理のサイドチェーン・スムージング値を選択。",
        "スペクトラム処理のサイドチェーン・ティルトスロープを調整。",
        "スペクトラム処理のサイドチェーン・アタック・スキューを調整。スキュー値が高くなるほど、高周波数帯のアタックが速くなります。",
        "スペクトラム処理のサイドチェーン・リリース・スキューを調整。スキュー値が高くなるほど、高周波数帯のリリースのスピードが速くなります。",
        "相対ラウドネスのスペクトラム・ゲートを調整。",

        "追加の出力ゲインを調整。",
        "すべてのフィルターのベース＆ターゲットゲインのスケールを調整。",
        "押す: 静的ゲイン補正（SGC）をオン。SGCは正確ではありませんが、ダイナミクスには影響しません。",
        "押す: 入力信号と出力信号のインテグレーテッド・ラウドネスの計測を開始\n離す: 出力ゲインを2つのラウドネス値の差分に更新。",

        "押す: 入力信号のスペクトラム・アナライザーをオン",
        "押す: 出力信号のスペクトラム・アナライザーをオン",
        "押す: サイドチェーン信号のスペクトラム・アナライザーをオン",
        "スペクトラム・アナライザーのディケイ速度を選択。",
        "スペクトラム・アナライザーのティルトスロープを選択。",
        "スペクトラム・アナライザーのスムージング値を選択。",
        "スペクトラム・アナライザーのスムージングタイプを選択。",
        "押す: フリーズ機能をオン。アナライザー上にマウスカーソルを重ねるとスペクトラムがフリーズします。",
        "押す: 衝突検出をオン。",
        "衝突検出の強度を調整。",

        "ダブルクリック: UI設定を開く。",
    };
}
