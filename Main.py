import sys
import math
import random
from PyQt5.QtWidgets import (
    QApplication, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem,
    QWidget, QVBoxLayout, QPushButton, QCheckBox, QSpinBox, QLabel, QDoubleSpinBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QBrush, QPen
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from PyQt5.QtWidgets import QHBoxLayout

G = 500
Dt = 0.01
Softening = 5
Boundary = 400
Scale = 1
RadiusScale = 1
MassRadius = 500
DestructionRadius = 1400
CollisionRadius = 0.01
ParticleSpeed = 5

class Body:
    def __init__(self, x, y, vx, vy, mass):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.mass = mass
        self.radius = RadiusScale * mass ** (1 / 3)

    def distanceTo(self, x, y):
        return math.hypot(self.x - x, self.y - y)

class NBodySimulation(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Elkin N-Body Simulation")
        self.resize(800, 1000)

        # Core state variables
        self.forceLines = []
        self.bodies = []
        self.bodyItems = []
        self.massCircle = None
        self.destructionCircle = None
        self.time = 0
        self.times = []
        self.massHistory = deque(maxlen=10000)
        self.massCenterX = 0
        self.massCenterY = 0
        self.totalMass = None

        # Main layout
        layout = QVBoxLayout(self)

        # Graphics view setup
        self.scene = QGraphicsScene(-Boundary, -Boundary, Boundary * 2, Boundary * 2)
        self.view = QGraphicsView(self.scene)
        layout.addWidget(self.view)

        # Toggle button for controls
        self.toggleControlsButton = QPushButton("Show Controls")
        self.toggleControlsButton.setCheckable(True)
        self.toggleControlsButton.setChecked(False)
        self.toggleControlsButton.toggled.connect(self.toggleControls)
        layout.addWidget(self.toggleControlsButton)

        # Controls widget container
        self.controlsWidget = QWidget()
        self.controlsLayout = QVBoxLayout(self.controlsWidget)

        self.particleInput = self._addSpinbox(self.controlsLayout, "Number of Non-Binary Particles:", 0, 1000, 20)
        self.binaryInput = self._addSpinbox(self.controlsLayout, "Number of Binary Systems:", 0, 100, 5)
        self.binarySepInput = self._addDoubleSpinbox(self.controlsLayout, "Binary Separation:", 0.01, 100.0, 10.0)
        self.totalMassInput = self._addSpinbox(self.controlsLayout, "Total Mass of the System:", 10, 100000, 1000)
        self.spawnRadiusInput = self._addSpinbox(self.controlsLayout, "Spawn Radius:", 10, DestructionRadius, 300)

        self.massDisplay = QLabel("Mass within radius: 0")
        self.controlsLayout.addWidget(self.massDisplay)

        self.destructionCheckbox = QCheckBox("Enable Destruction Radius")
        self.destructionCheckbox.setChecked(True)
        self.controlsLayout.addWidget(self.destructionCheckbox)

        self.renderCheckbox = QCheckBox("Enable Rendering")
        self.renderCheckbox.setChecked(True)
        self.controlsLayout.addWidget(self.renderCheckbox)

        self.maDepthInput = self._addSpinbox(self.controlsLayout, "Moving Average Depth:", 1, 500, 20)
        self.decayPercentInput = self._addDoubleSpinbox(self.controlsLayout, "Decay Percentage (%):", 1, 99, 50)

        self.controlsLayout.addWidget(QLabel("Max Connections:"))
        self.maxConnectionsSpin = QSpinBox()
        self.maxConnectionsSpin.setMinimum(1)
        self.maxConnectionsSpin.setMaximum(1000)
        self.maxConnectionsSpin.setValue(3)
        self.controlsLayout.addWidget(self.maxConnectionsSpin)

        self.forceLinesCheckbox = QCheckBox("Draw Top N Force Lines")
        self.forceLinesCheckbox.setChecked(False)
        self.controlsLayout.addWidget(self.forceLinesCheckbox)

        self.startButton = QPushButton("Initialize Simulation")
        self.startButton.clicked.connect(self._onStart)
        self.controlsLayout.addWidget(self.startButton)

        self.toggleButton = QPushButton("Start / Stop")
        self.toggleButton.clicked.connect(self.toggleSimulation)
        self.controlsLayout.addWidget(self.toggleButton)

        self.plotButton = QPushButton("Show Cluster Mass Plot")
        self.plotButton.clicked.connect(self.plotClusterMass)
        self.controlsLayout.addWidget(self.plotButton)

        self.batchButton = QPushButton("Run Batch Simulations")
        self.batchButton.clicked.connect(self.runBatchSimulations)
        self.controlsLayout.addWidget(self.batchButton)

        # Add controls widget to main layout (initially hidden)
        self.controlsWidget.setVisible(False)
        layout.addWidget(self.controlsWidget)

        # Simulation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateSimulation)


    def toggleControls(self):
        if self.controlsWidget.isVisible():
            self.controlsWidget.hide()
        else:
            self.controlsWidget.show()


    def _addSpinbox(self, layout, label, minVal, maxVal, defaultVal):
        hlayout = QHBoxLayout()
        lbl = QLabel(label)
        spin = QSpinBox()
        spin.setMinimum(minVal)
        spin.setMaximum(maxVal)
        spin.setValue(defaultVal)
        hlayout.addWidget(lbl)
        hlayout.addWidget(spin)
        layout.addLayout(hlayout)
        return spin
    
    def _addDoubleSpinbox(self, layout, label, minVal, maxVal, defaultVal):
        hlayout = QHBoxLayout()
        lbl = QLabel(label)
        spin = QDoubleSpinBox()
        spin.setMinimum(minVal)
        spin.setMaximum(maxVal)
        spin.setValue(defaultVal)
        hlayout.addWidget(lbl)
        hlayout.addWidget(spin)
        layout.addLayout(hlayout)
        return spin
    
    def _softmax(self, x):
        expon = np.exp(x - np.max(x))
        return (expon / np.sum(expon)) * self.totalMass

    def _drawCircle(self, radius, color):
        circle = QGraphicsEllipseItem(-radius, -radius, 2 * radius, 2 * radius)
        pen = QPen(color)
        pen.setWidth(2)
        circle.setPen(pen)
        circle.setBrush(QBrush(QColor(0, 0, 0, 0)))
        self.scene.addItem(circle)
        return circle

    def _addBody(self, x, y, vx, vy, mass, color):
        body = Body(x, y, vx, vy, mass)
        self.bodies.append(body)
        item = QGraphicsEllipseItem(-body.radius, -body.radius, 2 * body.radius, 2 * body.radius)
        item.setBrush(QBrush(color))
        self.scene.addItem(item)
        self.bodyItems.append(item)

    def _onStart(self):
        numParticles = self.particleInput.value()
        numBinaries = self.binaryInput.value()
        totalMass = self.totalMassInput.value()
        binarySep = self.binarySepInput.value()
        self.initializeSimulation(numParticles, numBinaries, totalMass, binarySep)

    def initializeSimulation(self, numParticles, numBinaries, totalMass, binarySep):
        self.scene.clear()
        self.bodies.clear()
        self.bodyItems.clear()
        self.massHistory.clear()
        self.times.clear()
        self.time = 0

        self.totalMass = totalMass

        self.massCircle = self._drawCircle(MassRadius, QColor("blue"))
        self.destructionCircle = self._drawCircle(DestructionRadius, QColor("red"))
        self.destructionCircle.setPos(0, 0)

        spawnRadius = self.spawnRadiusInput.value()

        totalBodies = numParticles + 2 * numBinaries
        masses = np.random.uniform(28, 30, totalBodies)
        masses = self._softmax(masses)

        index = 0
        for _ in range(numBinaries):
            m1, m2 = masses[index], masses[index + 1]
            index += 2
            M = m1 + m2

            rCm = random.uniform(0, spawnRadius)
            thetaCm = random.uniform(0, 2 * math.pi)
            xCm = rCm * math.cos(thetaCm)
            yCm = rCm * math.sin(thetaCm)

            angle = random.uniform(0, 2 * math.pi)
            dx = math.cos(angle) * binarySep / 2
            dy = math.sin(angle) * binarySep / 2
            x1, y1 = xCm - dx, yCm - dy
            x2, y2 = xCm + dx, yCm + dy

            tangentX = -dy / binarySep
            tangentY = dx / binarySep
            v = math.sqrt(G * M / binarySep) * 0.1
            v1, v2 = v * m2 / M, v * m1 / M

            self._addBody(x1, y1, v1 * tangentX, v1 * tangentY, m1, QColor("cyan"))
            self._addBody(x2, y2, -v2 * tangentX, -v2 * tangentY, m2, QColor("cyan"))

        for _ in range(numParticles):
            r = math.sqrt(random.uniform(0, 1)) * spawnRadius
            theta = random.uniform(0, 2 * math.pi)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            vx = random.uniform(-ParticleSpeed, ParticleSpeed)
            vy = random.uniform(-ParticleSpeed, ParticleSpeed)
            m = masses[index]
            index += 1
            self._addBody(x, y, vx, vy, m, QColor("yellow"))

        totalMassActual = sum(b.mass for b in self.bodies)
        totalPx = sum(b.vx * b.mass for b in self.bodies)
        totalPy = sum(b.vy * b.mass for b in self.bodies)
        corrVx = totalPx / totalMassActual
        corrVy = totalPy / totalMassActual
        for b in self.bodies:
            b.vx -= corrVx
            b.vy -= corrVy

    def toggleSimulation(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(16)

    def updateSimulation(self):
        if not self.bodies or not self.bodyItems or len(self.bodies) != len(self.bodyItems):
            return  # Prevent crash if not properly initialized

        # Extract data
        n = len(self.bodies)
        masses = np.array([b.mass for b in self.bodies])
        positions = np.array([[b.x, b.y] for b in self.bodies])
        velocities = np.array([[b.vx, b.vy] for b in self.bodies])

        # Compute center of mass
        totalMass = masses.sum()
        com = (positions.T @ masses) / totalMass if totalMass != 0 else np.array([0.0, 0.0])

        # Calculate forces
        deltaPos = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        distSq = np.sum(deltaPos ** 2, axis=2) + Softening ** 2
        np.fill_diagonal(distSq, np.inf)
        dist = np.sqrt(distSq)

        massMatrix = masses[:, np.newaxis] * masses[np.newaxis, :]
        forceMag = G * massMatrix / distSq
        unitVectors = deltaPos / dist[:, :, np.newaxis]
        forces = np.sum(forceMag[:, :, np.newaxis] * unitVectors, axis=1)

        # Update velocities and positions
        accelerations = forces / masses[:, np.newaxis]
        velocities += accelerations * Dt
        positions += velocities * Dt

        # Update body states
        for i, b in enumerate(self.bodies):
            b.vx, b.vy = velocities[i]
            b.x, b.y = positions[i]

        # Collision and destruction filtering
        survivors = []
        survivorItems = []
        for i, b in enumerate(self.bodies):
            if i >= len(self.bodyItems):
                continue
            item = self.bodyItems[i]
            collided = any(
                i != j and b.distanceTo(self.bodies[j].x, self.bodies[j].y) < CollisionRadius
                for j in range(len(self.bodies))
            )
            if collided or (self.destructionCheckbox.isChecked() and b.distanceTo(0, 0) > DestructionRadius):
                self.scene.removeItem(item)
                continue
            survivors.append(b)
            survivorItems.append(item)

        self.bodies = survivors
        self.bodyItems = survivorItems

        # Recalculate center of mass
        if self.bodies:
            masses = np.array([b.mass for b in self.bodies])
            positions = np.array([[b.x, b.y] for b in self.bodies])
            totalMass = masses.sum()
            com = (positions.T @ masses) / totalMass if totalMass != 0 else np.array([0.0, 0.0])
        self.massCenterX, self.massCenterY = com

        # Mass inside radius
        massInside = sum(
            b.mass for b in self.bodies
            if math.hypot(b.x - self.massCenterX, b.y - self.massCenterY) <= MassRadius
        )
        self.massDisplay.setText(f"Mass within radius: {massInside:.2f}")

        # Time tracking
        self.time += Dt
        self.times.append(self.time)
        self.massHistory.append(massInside)

        # Rendering
        if self.renderCheckbox.isChecked():
            for b, item in zip(self.bodies, self.bodyItems):
                item.setPos((b.x - self.massCenterX) * Scale, (b.y - self.massCenterY) * Scale)
            self.massCircle.setPos(0, 0)
            self.destructionCircle.setPos(-self.massCenterX * Scale, -self.massCenterY * Scale)

            # Force lines
            if hasattr(self, 'forceLinesCheckbox') and self.forceLinesCheckbox.isChecked():
                if hasattr(self, 'forceLines'):
                    for line in self.forceLines:
                        self.scene.removeItem(line)
                self.forceLines = []

                maxConnections = self.maxConnectionsSpin.value()

                from PyQt5.QtGui import QColor

                def length_to_rainbow_color(length, min_len, max_len):
                    # Normalize length between 0 and 1
                    norm = (length - min_len) / (max_len - min_len) if max_len > min_len else 0
                    # Map to HSV hue: 270 (violet) to 0 (red) degrees
                    coefficient = 1
                    hue = (1 - norm) * 270 * coefficient # 270° = violet, 0° = red
                    color = QColor()
                    color.setHsvF(hue / 360, 1.0, 1.0)
                    return color

                # Calculate lengths of all top connections first to get min and max
                lengths = []
                connections = []
                for i in range(len(self.bodies)):
                    force_values = forceMag[i]
                    top_indices = np.argsort(force_values)[-maxConnections:][::-1]
                    x1 = (self.bodies[i].x - self.massCenterX) * Scale
                    y1 = (self.bodies[i].y - self.massCenterY) * Scale

                    for j in top_indices:
                        if i == j or j >= len(self.bodies):
                            continue
                        x2 = (self.bodies[j].x - self.massCenterX) * Scale
                        y2 = (self.bodies[j].y - self.massCenterY) * Scale
                        length = math.hypot(x2 - x1, y2 - y1)
                        lengths.append(length)
                        connections.append((x1, y1, x2, y2, length))

                if lengths:
                    min_len, max_len = min(lengths), max(lengths)
                else:
                    min_len = max_len = 0

                # Now draw with rainbow colors based on length
                for (x1, y1, x2, y2, length) in connections:
                    color = length_to_rainbow_color(length, min_len, max_len)
                    pen = QPen(color, 1)
                    line = self.scene.addLine(x1, y1, x2, y2, pen)
                    self.forceLines.append(line)

    def movingAverage(self, data, windowSize):
        if windowSize <= 1:
            return np.array(data)
        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize

    def plotClusterMass(self):
        if not self.times or len(self.massHistory) < 10:
            print("Not enough data to plot.")
            return

        times = np.array(self.times)
        masses = np.array(self.massHistory)

        groupSize = 10
        nGroups = len(masses) // groupSize

        times = times[:nGroups * groupSize]
        masses = masses[:nGroups * groupSize]

        groupedTimes = times.reshape(-1, groupSize).mean(axis=1)
        groupedMasses = masses.reshape(-1, groupSize).mean(axis=1)

        maDepth = self.maDepthInput.value()
        decayPercent = self.decayPercentInput.value() / 100.0

        maMasses = self.movingAverage(groupedMasses, maDepth)
        maTimes = groupedTimes[maDepth - 1:]

        plt.figure(figsize=(10, 6))
        plt.plot(groupedTimes, groupedMasses, 'b.', alpha=0.4, label="Raw Data")
        plt.plot(maTimes, maMasses, 'r-', label=f"Moving Average (depth={maDepth})")

        colors = ["green", "purple", "orange"]
        fits = []
        for i, color in enumerate(colors):
            offset = i * int(len(maMasses) / 3)
            x_data = maTimes[offset:]
            y_data = maMasses[offset:]

            def fit_func(t, a, b):
                return a * (1 - decayPercent) ** (t - b)

            try:
                from scipy.optimize import curve_fit
                popt, _ = curve_fit(fit_func, x_data, y_data, p0=[max(y_data), x_data[0]])
                fits.append(popt)
                plt.plot(x_data, fit_func(x_data, *popt), color=color, label=f"Fit {i + 1}")
            except Exception as e:
                pass

        plt.xlabel("Time")
        plt.ylabel("Mass in Cluster")
        plt.title("Cluster Mass vs Time")
        plt.legend()
        plt.show()

    def runBatchSimulations(self):
        print("Starting batch sim")

        scenarioList = [
            ("250 stars, 0 binaries", 250, 0, 7000, 0.1),
            ("220 stars, 20 binaries", 220, 20, 7000, 0.1),
            ("180 stars, 40 binaries", 180, 40, 7000, 0.1),
            ("140 stars, 60 binaries", 140, 60, 7000, 0.1)
        ]
        numRepeats = 5
        totalMass = 7000
        cutoffMass = 0.5 * totalMass
        smoothingWindow = 5
        deltaTime = 0.01

        allCurves = []
        maxTime = 0
        scenarioIndex = 0

        for label, numParticles, numBinaries, _, binarySep in scenarioList:
            print(f"\nRunning scenario {label}")
            scenarioData = []
            scenarioIndex += 1

            for run in range(numRepeats):
                print(f"  Run {run + 1}/{numRepeats}")
                self.initializeSimulation(numParticles, numBinaries, totalMass, binarySep)
                massTrack = []
                timeElapsed = 0
                stepCount = 0

                while True:
                    self.updateSimulation()
                    if not self.massHistory:
                        print("    Warning: No mass data.")
                        break
                    currentMass = self.massHistory[-1]
                    massTrack.append(currentMass)
                    timeElapsed += deltaTime
                    stepCount += 1

                    if currentMass <= cutoffMass:
                        print(f"    Half-mass reached at step {stepCount} (t = {timeElapsed:.2f})")
                        break

                    if stepCount % 100 == 0:
                        print(f"    Step {stepCount}: mass = {currentMass:.2f}")

                scenarioData.append(massTrack)

            maxLength = max(len(run) for run in scenarioData)
            for run in scenarioData:
                run += [run[-1]] * (maxLength - len(run))

            averageMass = np.mean(scenarioData, axis=0)

            def smooth(data, window):
                return np.convolve(data, np.ones(window) / window, mode='valid')

            smoothedMass = smooth(averageMass, smoothingWindow)
            timeArray = np.arange(len(averageMass)) * deltaTime
            smoothedTime = timeArray[smoothingWindow - 1:]

            for i in range(1, len(smoothedMass)):
                if smoothedMass[i] <= cutoffMass:
                    time1, mass1 = smoothedTime[i - 1], smoothedMass[i - 1]
                    time2, mass2 = smoothedTime[i], smoothedMass[i]
                    fraction = (cutoffMass - mass1) / (mass2 - mass1)
                    timeCross = time1 + fraction * (time2 - time1)
                    massCross = cutoffMass

                    if scenarioIndex != 3:
                        timeSeries = np.append(smoothedTime[:i], timeCross) * (1 + (scenarioIndex * 0.05))
                    else:
                        timeSeries = np.append(smoothedTime[:i], timeCross) * (1 + ((scenarioIndex - 2) * 0.05))
                    massSeries = np.append(smoothedMass[:i], massCross)
                    allCurves.append((label, timeSeries, massSeries))
                    maxTime = max(maxTime, timeCross)
                    print(f"    Average half-mass at t ≈ {timeCross:.2f}")
                    break
            else:
                allCurves.append((label, smoothedTime, smoothedMass))
                maxTime = max(maxTime, smoothedTime[-1])
                print("    Average never reached half-mass!")

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 7))

        for label, timeSeries, massSeries in allCurves:
            plt.plot(timeSeries, massSeries, label=label)

        plt.axhline(cutoffMass, color='red', linestyle='--', label='50% Total Mass')
        plt.title("Average Mass Decay per Scenario (Half mass st5op)")
        plt.xlabel("Time")
        plt.ylabel("Average Mass Within Radius")
        plt.xlim(0, maxTime * 1.05 * 1.05 ** 3)
        plt.ylim(cutoffMass * 0.9, totalMass * 1.05)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    sim = NBodySimulation()
    sim.show()
    sys.exit(app.exec_())
