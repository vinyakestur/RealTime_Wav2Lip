//
//  MouthShape.swift
//  Frontend_wav2lip
//

import SwiftUI

enum MouthShape: String, CaseIterable {
    case neutral
    case openWide
    case closed
    case rounded
    case wide
    
    static func fromPrediction(_ value: Int) -> MouthShape {
        switch value {
        case 0:
            return .neutral
        case 1:
            return .openWide
        case 2:
            return .closed
        case 3:
            return .rounded
        case 4:
            return .wide
        default:
            return .neutral
        }
    }
}
