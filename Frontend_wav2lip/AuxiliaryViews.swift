import SwiftUI

// MARK: - MouthAnimationView
struct MouthAnimationView: View {
    let currentShape: MouthShape
    
    var body: some View {
        Canvas { context, size in
            let rect = CGRect(origin: .zero, size: size)
            
            switch currentShape {
            case .neutral:
                drawNeutralMouth(in: context, rect: rect)
            case .openWide:
                drawOpenWideMouth(in: context, rect: rect)
            case .closed:
                drawClosedMouth(in: context, rect: rect)
            case .rounded:
                drawRoundedMouth(in: context, rect: rect)
            case .wide:
                drawWideMouth(in: context, rect: rect)
            }
        }
    }
    
    private func drawNeutralMouth(in context: GraphicsContext, rect: CGRect) {
        var path = Path()
        path.move(to: CGPoint(x: rect.minX, y: rect.midY))
        path.addCurve(
            to: CGPoint(x: rect.maxX, y: rect.midY),
            control1: CGPoint(x: rect.width * 0.3, y: rect.midY + 3),
            control2: CGPoint(x: rect.width * 0.7, y: rect.midY + 3)
        )
        context.stroke(path, with: .color(.black), lineWidth: 2)
    }
    
    private func drawOpenWideMouth(in context: GraphicsContext, rect: CGRect) {
        var path = Path()
        path.addEllipse(in: rect.insetBy(dx: 3, dy: 3))
        context.fill(path, with: .color(.black))
    }
    
    private func drawClosedMouth(in context: GraphicsContext, rect: CGRect) {
        var path = Path()
        path.move(to: CGPoint(x: rect.minX, y: rect.midY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.midY))
        context.stroke(path, with: .color(.black), lineWidth: 2)
    }
    
    private func drawRoundedMouth(in context: GraphicsContext, rect: CGRect) {
        var path = Path()
        path.addEllipse(in: rect.insetBy(dx: 12, dy: 8))
        context.stroke(path, with: .color(.black), lineWidth: 2)
    }
    
    private func drawWideMouth(in context: GraphicsContext, rect: CGRect) {
        var path = Path()
        path.move(to: CGPoint(x: rect.minX, y: rect.midY))
        path.addCurve(
            to: CGPoint(x: rect.maxX, y: rect.midY),
            control1: CGPoint(x: rect.width * 0.3, y: rect.midY - 8),
            control2: CGPoint(x: rect.width * 0.7, y: rect.midY - 8)
        )
        context.stroke(path, with: .color(.black), lineWidth: 2)
    }
}

// MARK: - AudioWaveformView
struct AudioWaveformView: View {
    let amplitude: CGFloat
    
    var body: some View {
        GeometryReader { geometry in
            Path { path in
                let width = geometry.size.width
                let height = geometry.size.height
                let midPoint = height / 2
                
                path.move(to: CGPoint(x: 0, y: midPoint))
                
                for x in stride(from: 0, to: width, by: 2) {
                    let y = midPoint + amplitude * sin(x/8) * height/4
                    path.addLine(to: CGPoint(x: x, y: y))
                }
            }
            .stroke(Color.blue, lineWidth: 1.5)
        }
    }
}
