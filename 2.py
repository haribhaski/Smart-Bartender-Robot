import pybullet as p
import pybullet_data
import numpy as np
import time
import json
import os
from scipy import linalg
from scipy.interpolate import CubicSpline
import threading

class NovelPathPlanner:
    """Enhanced path planner with proper obstacle avoidance"""
    def __init__(self, obstacles=None, bounds=[-5, 5, -5, 5]):
        self.obstacles = obstacles or []
        self.bounds = bounds
        self.rho = 1.5
        self.alpha = 1.2
        self.max_iter = 50
        self.tolerance = 1e-4
        self.min_height = 0.2
        self.max_height = 0.6
        
    def plan_path(self, start, goal, num_waypoints=30):
        """Generate smooth path avoiding obstacles"""
        path = np.linspace(start, goal, num_waypoints)
        
        # ADMM variables
        z = path.copy()
        u = np.zeros_like(path)
        
        for iteration in range(self.max_iter):
            # Path update with smoothness constraints
            path = self._update_path(path, z, u, start, goal)
            
            # Obstacle projection
            z_prev = z.copy()
            z = self._update_z(path, u)
            
            # Dual variable update
            u = u + path - z
            
            # Check convergence
            if (np.linalg.norm(path - z) < self.tolerance and 
                np.linalg.norm(z - z_prev) < self.tolerance):
                break
        
        return self._smooth_path(path)
    
    def _update_path(self, path, z, u, start, goal):
        """Update path for smoothness while maintaining constraints"""
        n = len(path)
        A = np.zeros((n-2, n))
        for i in range(n-2):
            A[i, i] = 1
            A[i, i+1] = -2
            A[i, i+2] = 1
        
        # Construct quadratic term matrix
        H = 0.1 * A.T @ A + self.rho * np.eye(n)
        
        # Construct linear term
        q = self.rho * (z - u)
        
        # Solve for each dimension with endpoint constraints
        new_path = np.zeros_like(path)
        for dim in range(3):
            # Apply endpoint constraints
            M = H.copy()
            M[0,:] = 0; M[0,0] = 1  # Fix start position
            M[-1,:] = 0; M[-1,-1] = 1  # Fix goal position
            
            b = q[:,dim].copy()
            b[0] = start[dim]
            b[-1] = goal[dim]
            
            new_path[:,dim] = np.linalg.solve(M, b)
        
        return new_path
    
    def _update_z(self, path, u):
        """Project path points to obstacle-free space"""
        z = path + u
        
        for i in range(1, len(z)-1):  # Keep start and goal fixed
            point = z[i]
            
            # Enforce workspace bounds and height limits
            point[0] = np.clip(point[0], self.bounds[0], self.bounds[1])
            point[1] = np.clip(point[1], self.bounds[2], self.bounds[3])
            point[2] = np.clip(point[2], self.min_height, self.max_height)
            
            # Handle obstacles
            for obs in self.obstacles:
                obs_pos = np.array(obs[:3])
                obs_radius = obs[3] + 0.25  # Safety margin
                
                vec = point[:2] - obs_pos[:2]
                dist = np.linalg.norm(vec)
                
                if dist < obs_radius:
                    if dist < 1e-6:  # Avoid division by zero
                        vec = np.array([1, 0])  # Default push direction
                        dist = 1.0
                    
                    # Push point outside obstacle radius
                    push_distance = obs_radius - dist
                    push_vector = (vec / dist) * push_distance * 1.1
                    
                    z[i,:2] += push_vector
        
        return z
    
    def _smooth_path(self, path):
        """Apply cubic spline smoothing"""
        t = np.linspace(0, 1, len(path))
        cs = [CubicSpline(t, path[:,i]) for i in range(3)]
        t_fine = np.linspace(0, 1, len(path)*2)
        return np.column_stack([c(t_fine) for c in cs])
    
    def update_obstacles(self, new_obstacles):
        """Update obstacle positions and sizes"""
        self.obstacles = new_obstacles   
        
class LQRController:
    """LQR controller for balancing juice while moving"""
    def __init__(self):
        self.dt = 0.01  # Control timestep
        
        # System dynamics: [x, y, θx, θy, dx, dy, dθx, dθy]
        self.A = np.array([
            [1, 0, 0, 0, self.dt, 0, 0, 0],          # x position
            [0, 1, 0, 0, 0, self.dt, 0, 0],          # y position
            [0, 0, 1, 0, 0, 0, self.dt, 0],          # θx angle
            [0, 0, 0, 1, 0, 0, 0, self.dt],          # θy angle
            [0, 0, 0, 0, 1, 0, 0, 0],                # x velocity
            [0, 0, 0, 0, 0, 1, 0, 0],                # y velocity
            [0, 0, -9.81*self.dt*0.5, 0, 0, 0, 1, 0], # θx velocity with gravity
            [0, 0, 0, -9.81*self.dt*0.5, 0, 0, 0, 1]  # θy velocity with gravity
        ])
        
        # Control input matrix: [Fx, Fy, τx, τy]
        self.B = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [self.dt, 0, 0, 0],
            [0, self.dt, 0, 0],
            [0, 0, self.dt*0.5, 0],
            [0, 0, 0, self.dt*0.5]
        ])
        
        # Cost matrices
        self.Q = np.diag([2.0, 2.0, 200.0, 200.0, 2.0, 2.0, 20.0, 20.0])
        self.R = np.diag([0.05, 0.05, 0.5, 0.5])
        
        # Compute LQR gain matrix
        self.K = self._compute_lqr_gain()
        self.state = np.zeros(8)
        
    def _compute_lqr_gain(self):
        """Compute the optimal LQR gain matrix"""
        P = linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
        K = np.linalg.inv(self.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
        return K
    
    def update(self, position, velocity, acceleration):
        """Compute control action to balance juice"""
        current_tilt = [0, 0]
        self.state = np.array([
            position[0], position[1],
            current_tilt[0], current_tilt[1],
            velocity[0], velocity[1],
            0, 0
        ])
        u = -self.K @ self.state
        force_adjustment = u[:2]
        torque_adjustment = u[2:4]
        modified_acceleration = np.array([
            acceleration[0] + force_adjustment[0] * 0.3,
            acceleration[1] + force_adjustment[1] * 0.3,
            acceleration[2]
        ])
        return modified_acceleration


class JuiceParticleSystem:
    """Visual representation of juice with physics-based behavior"""
    def __init__(self, glass_id, color=[1.0, 0.5, 0.0, 0.8]):
        self.glass_id = glass_id
        self.color = color
        self.particles = []
        self.particle_radius = 0.01
        self.fill_level = 0.1
        self.is_active = False
        self.damping = 0.9
        self.spring_constant = 250.0
        self.gravity = -9.81
        self.particle_mass = 0.001
        self.last_time = time.time()
        self.rest_height = 0.04
    
    def update(self, acceleration):
        """Update juice particle positions based on glass movement"""
        if not self.is_active or len(self.particles) == 0:
            return
            
        glass_pos, glass_orn = p.getBasePositionAndOrientation(self.glass_id)
        glass_bottom = [glass_pos[0], glass_pos[1], glass_pos[2] - 0.07]
        glass_top = [glass_pos[0], glass_pos[1], glass_pos[2] + 0.07]
        
        tilt_angle_x = min(max(-0.2 * acceleration[1], -0.3), 0.3)
        tilt_angle_y = min(max(0.2 * acceleration[0], -0.3), 0.3)
        
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        valid_particles = []
        for i, particle_id in enumerate(self.particles):
            try:
                old_pos = p.getBasePositionAndOrientation(particle_id)[0]
                new_x = glass_pos[0] + 0.03 * np.sin(tilt_angle_y) * (old_pos[2] - glass_bottom[2]) / 0.15
                new_y = glass_pos[1] - 0.03 * np.sin(tilt_angle_x) * (old_pos[2] - glass_bottom[2]) / 0.15
                
                dx = new_x - glass_pos[0]
                dy = new_y - glass_pos[1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 0.04:
                    scale = 0.04 / dist
                    new_x = glass_pos[0] + dx * scale
                    new_y = glass_pos[1] + dy * scale
                    
                new_z = glass_bottom[2] + self.rest_height
                new_x += np.random.uniform(-0.001, 0.001)
                new_y += np.random.uniform(-0.001, 0.001)
                new_z += np.random.uniform(-0.001, 0.001)
                
                if new_z > glass_top[2] - 0.02:
                    new_z = glass_top[2] - 0.02
                    
                p.resetBasePositionAndOrientation(
                    particle_id,
                    [new_x, new_y, new_z],
                    p.getQuaternionFromEuler([0, 0, 0])
                )
                
                valid_particles.append(particle_id)
            except Exception as e:
                print(f"Error updating particle {particle_id}: {e}")
                
        self.particles = valid_particles
        
    def activate(self, active=True):
        """Toggle juice visibility"""
        self.is_active = active
        try:
            self._create_particles()
        except Exception as e:
            print(f"Error creating particles: {e}")
            self.is_active = False
            self.particles = []
            
    def _create_particles(self):
        """Initialize or refresh juice particles"""
        for p_id in self.particles:
            try:
                p.removeBody(p_id)
            except Exception as e:
                print(f"Error removing particle {p_id}: {e}")
        self.particles = []
        
        if not self.is_active:
            return
            
        num_particles = 30
        try:
            glass_pos, glass_orn = p.getBasePositionAndOrientation(self.glass_id)
            glass_bottom = [glass_pos[0], glass_pos[1], glass_pos[2] - 0.075]
            
            for i in range(num_particles):
                r = np.random.uniform(0, 0.04)
                theta = np.random.uniform(0, 2*np.pi)
                height = np.random.uniform(0, self.fill_level)
                pos = [
                    glass_bottom[0] + r * np.cos(theta),
                    glass_bottom[1] + r * np.sin(theta),
                    glass_bottom[2] + height
                ]
                visualId = p.createVisualShape(p.GEOM_SPHERE, radius=self.particle_radius, rgbaColor=self.color)
                particleId = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visualId,
                    basePosition=pos
                )
                self.particles.append(particleId)
        except Exception as e:
            print(f"Error in particle creation: {e}")
            self.is_active = False
            self.particles = []


class ImageBasedObstacleDetection:
    """Obstacle detection using camera images"""
    def __init__(self, width=320, height=240):
        self.width = width
        self.height = height
        self.view_matrix = p.computeViewMatrix(
            cameraEyePosition=[5, 5, 5],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 0, 1]
        )
        self.projection_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width)/height,
            nearVal=0.1,
            farVal=100.0
        )
        self.last_detected_obstacles = []
        
    def capture_image(self):
        """Capture RGB and depth images from camera"""
        img_data = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_img = np.reshape(img_data[2], (self.height, self.width, 4))[:,:,:3]
        depth_img = np.reshape(img_data[3], (self.height, self.width))
        return rgb_img, depth_img
    
    def detect_obstacles(self):
        """Detect obstacles from camera image"""
        rgb_img, depth_img = self.capture_image()
        detected_obstacles = []
        
        for obs in OBSTACLES:  # Using global obstacles for simulation
            obj_pos = np.array(obs[:3])
            obj_size = obs[3]
            detected_obstacles.append({
                'position': obj_pos,
                'size': obj_size
            })
            
        self.last_detected_obstacles = detected_obstacles
        return detected_obstacles
    
    def update_planner_obstacles(self, planner):
        """Update path planner with detected obstacles"""
        obstacles = self.detect_obstacles()
        planner.update_obstacles([[o['position'][0], o['position'][1], o['position'][2], o['size']] 
                                 for o in obstacles])


class OrderManager:
    """Manages juice orders from data.json"""
    def __init__(self):
        self.orders = []
        self.current_order = None
        
    def load_orders(self):
        """Load orders from data.json file"""
        try:
            with open("data.json", "r") as f:
                data = json.load(f)
                self.orders = data.get("orders", [])
                for i, order in enumerate(self.orders):
                    if "id" not in order:
                        order["id"] = i + 1
        except Exception as e:
            print(f"Error loading orders: {e}")
            self.orders = []
        
    def get_next_order(self):
        """Get the next order in queue"""
        if self.orders:
            self.current_order = self.orders[0]
            return self.current_order
        return None
        
    def complete_current_order(self):
        """Mark current order as completed and remove from data.json"""
        if self.current_order:
            if self.orders:
                for i, order in enumerate(self.orders):
                    if (order.get("table") == self.current_order.get("table") and
                        order.get("juice") == self.current_order.get("juice")):
                        self.orders.pop(i)
                        break
                try:
                    with open("data.json", "w") as f:
                        json.dump({"orders": self.orders}, f)
                except Exception as e:
                    print(f"Error updating orders: {e}")
            self.current_order = None


class RobotController:
    """Controls robot movement and path following"""
    def __init__(self, robot_id, obstacles):
        self.robot_id = robot_id
        self.num_joints = p.getNumJoints(robot_id)
        print(f"Robot has {self.num_joints} joints")
        
        self.movable_joints = []
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            print(f"Joint {i}: {joint_name}, Type: {joint_type}")
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.movable_joints.append(i)
                lower_limit = joint_info[8]
                upper_limit = joint_info[9]
                if upper_limit < lower_limit:
                    p.changeDynamics(robot_id, i, jointLowerLimit=-3.14, jointUpperLimit=3.14)
        
        print(f"Using {len(self.movable_joints)} movable joints: {self.movable_joints}")
        
        self.end_effector_index = -1  # Use base for R2D2
        self.planner = NovelPathPlanner(obstacles)
        self.lqr = LQRController()
        self.max_velocity = 2.0
        self.max_force = 200.0
        self.empty_robot_speed = 1.5
        self.full_robot_speed = 0.8
        self.current_speed = self.empty_robot_speed
        self.path = None
        self.current_waypoint_idx = 0
        self.waypoint_threshold = 0.1
        self.carrying_juice = False
        self.acceleration = np.zeros(3)
        self.last_position = self.get_end_effector_position()
        self.last_time = time.time()
        self.last_velocity = np.zeros(3)
        
    def get_end_effector_position(self):
        """Get current position of robot end effector"""
        try:
            if self.end_effector_index == -1:
                result = p.getBasePositionAndOrientation(self.robot_id)
                pos = result[0]
            else:
                state = p.getLinkState(self.robot_id, self.end_effector_index)
                pos = state[0]
            return np.array(pos)
        except p.error as e:
            print(f"Error getting robot position: {e}")
            # Return a default position if the robot position can't be retrieved
            return np.array([0, 0, 0.5])
        
    def follow_waypoint(self, target, dt):
        """Move robot toward a single waypoint"""
        current_position = self.get_end_effector_position()
        
            
        current_time = time.time()
        elapsed = current_time - self.last_time
        if elapsed > 0:
            current_velocity = (current_position - self.last_position) / elapsed
            self.acceleration = (current_velocity - self.last_velocity) / elapsed
            self.last_velocity = current_velocity
            self.last_position = current_position
            self.last_time = current_time
        
        if self.carrying_juice:
            self.acceleration = self.lqr.update(
                current_position,
                self.last_velocity,
                self.acceleration
            )
        
        direction = np.array(target) - current_position
        distance = np.linalg.norm(direction)
        
        if distance < self.waypoint_threshold:
            print(f"Reached waypoint: {target}")
            return True
        
        if distance > 0:
            direction = direction / distance
        
        speed = self.current_speed
        direction = direction * speed
        
        target_position = current_position + direction * dt * 2.0
        target_position = np.clip(target_position, [-5, -5, 0.2], [5, 5, 1.0])
        
        if self.end_effector_index == -1:
            _, current_orn = p.getBasePositionAndOrientation(self.robot_id)
            
            if np.linalg.norm(direction[:2]) > 0.001:
                yaw = np.arctan2(direction[1], direction[0])
                target_orn = p.getQuaternionFromEuler([0, 0, yaw])
            else:
                target_orn = current_orn
            
            p.resetBasePositionAndOrientation(
                self.robot_id,
                [target_position[0], target_position[1], target_position[2]],
                target_orn
            )
            
            for joint_idx in self.movable_joints:
                joint_info = p.getJointInfo(self.robot_id, joint_idx)
                if 'wheel' in joint_info[1].decode('utf-8').lower():
                    wheel_velocity = 5.0
                    p.setJointMotorControl2(
                        bodyUniqueId=self.robot_id,
                        jointIndex=joint_idx,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=wheel_velocity,
                        force=10.0
                    )
            
            return False
        else:
            try:
                joint_positions = p.calculateInverseKinematics(
                    self.robot_id,
                    self.end_effector_index,
                    target_position,
                    maxNumIterations=100,
                    residualThreshold=1e-5
                )
                
                if joint_positions is None or len(joint_positions) == 0:
                    print(f"Warning: IK failed for target {target_position}")
                    return False
                
                for i, joint_idx in enumerate(self.movable_joints):
                    if i < len(joint_positions):
                        p.setJointMotorControl2(
                            bodyUniqueId=self.robot_id,
                            jointIndex=joint_idx,
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=joint_positions[i],
                            maxVelocity=self.max_velocity,
                            force=self.max_force
                        )
                
                print(f"Moving to target: {target_position}, Joints: {joint_positions[:len(self.movable_joints)]}")
                return False
            
            except Exception as e:
                print(f"Error in IK calculation: {e}")
                return False
        
    def follow_path(self, dt):
        """Follow the current path"""
        if self.path is None or len(self.path) == 0 or self.current_waypoint_idx >= len(self.path):
            return True
        
        target = self.path[self.current_waypoint_idx]
        reached = self.follow_waypoint(target, dt)
        if reached:
            self.current_waypoint_idx += 1
        
        return self.current_waypoint_idx >= len(self.path)
    
    def plan_path(self, start, goal):
        """Plan a new path from start to goal"""
        path = self.planner.plan_path(start, goal)
        
        if path is None or len(path) == 0:
            print("Warning: Path planner returned empty path")
            direct_path = np.zeros((5, 3))
            for i in range(5):
                t = i / 4.0
                direct_path[i] = [(1-t)*start[0] + t*goal[0],
                                (1-t)*start[1] + t*goal[1],
                                (1-t)*start[2] + t*goal[2]]
            path = direct_path
        
        self.path = path
        self.current_waypoint_idx = 0
        return path
    
    def set_carrying_juice(self, carrying):
        """Update robot state when carrying juice"""
        self.carrying_juice = carrying
        self.current_speed = self.full_robot_speed if carrying else self.empty_robot_speed

class RobotBartenderStateMachine:
    """Manages robot bartender states and transitions"""
    def __init__(self, robot_id, glass_id):
        self.robot_id = robot_id
        self.glass_id = glass_id
        self.constraint_id = -1
        self.juice_system = None
        self.robot_controller = RobotController(robot_id, OBSTACLES)
        self.order_manager = OrderManager()
        
        self.STATES = {
            "IDLE": self.state_idle,
            "MOVING_TO_JUICE": self.state_moving_to_juice,
            "PICKING_GLASS": self.state_picking_glass,
            "FILLING_GLASS": self.state_filling_glass,
            "MOVING_TO_TABLE": self.state_moving_to_table,
            "LOWERING_GLASS": self.state_lowering_glass,
            "DELIVERING_JUICE": self.state_delivering_juice,
            "MOVING_TO_HOME": self.state_returning_home
        }
        
        self.current_state = "IDLE"
        self.next_state = None
        self.state_start_time = time.time()
        self.current_table_index = 0
        self.reset_glass()
    
    def update(self, dt):
        """Update state machine"""
        state_function = self.STATES.get(self.current_state)
        if state_function:
            state_function(dt)
        
        if self.juice_system and self.juice_system['juice_height'] > 0:
            glass_pos, glass_orn = p.getBasePositionAndOrientation(self.glass_id)
            juice_height = self.juice_system['juice_height']
            juice_pos = [glass_pos[0], glass_pos[1], glass_pos[2] - self.juice_system['glass_height']/2 + juice_height/2]
            
            if hasattr(self.robot_controller, 'acceleration'):
                acc = self.robot_controller.acceleration
                tilt_x = min(max(-0.2 * acc[1], -0.3), 0.3)
                tilt_y = min(max(0.2 * acc[0], -0.3), 0.3)
                orn = p.getQuaternionFromEuler([tilt_x, tilt_y, 0])
            else:
                orn = glass_orn
            
            p.resetBasePositionAndOrientation(
                self.juice_system['juice_id'],
                juice_pos,
                orn
            )
        
        if self.next_state:
            print(f"Transitioning from {self.current_state} to {self.next_state}")
            self.current_state = self.next_state
            self.next_state = None
            self.state_start_time = time.time()

    def state_filling_glass(self, dt):
        """Filling glass with juice state"""
        if time.time() - self.state_start_time > 1.5:
            self.juice_system['juice_height'] = 0.1
            juice_radius = self.juice_system['juice_radius']
            glass_pos, _ = p.getBasePositionAndOrientation(self.glass_id)
            
            juice_visual_id = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=juice_radius,
                length=self.juice_system['juice_height'],
                rgbaColor=[1.0, 0.5, 0.0, 0.8]
            )
            juice_collision_id = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=juice_radius,
                height=self.juice_system['juice_height']
            )
            p.removeBody(self.juice_system['juice_id'])
            self.juice_system['juice_id'] = p.createMultiBody(
                baseMass=0.01,
                baseCollisionShapeIndex=juice_collision_id,
                baseVisualShapeIndex=juice_visual_id,
                basePosition=[glass_pos[0], glass_pos[1], glass_pos[2] - self.juice_system['glass_height']/2 + self.juice_system['juice_height']/2],
                baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
            )
            self.next_state = "MOVING_TO_TABLE"

    def detach_glass_from_robot(self):
        """Detach glass from robot"""
        if self.constraint_id != -1:
            p.removeConstraint(self.constraint_id)
            self.constraint_id = -1
        self.robot_controller.set_carrying_juice(False)
        if self.juice_system:
            self.juice_system['juice_height'] = 0.0
            p.removeBody(self.juice_system['juice_id'])
            self.juice_system['juice_id'] = p.createMultiBody(
                baseMass=0.01,
                baseCollisionShapeIndex=p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=self.juice_system['juice_radius'], height=0.0),
                baseVisualShapeIndex=p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=self.juice_system['juice_radius'], length=0.0, rgbaColor=[1.0, 0.5, 0.0, 0.8]),
                basePosition=[0, 0, 0]
            )
        
    def reset_glass(self):
        """Reset glass to juice center"""
        if self.constraint_id != -1:
            p.removeConstraint(self.constraint_id)
            self.constraint_id = -1
        if hasattr(self, 'glass_id') and self.glass_id != -1:
            p.removeBody(self.glass_id)
        self.glass_id, self.constraint_id, self.juice_system = create_glass(JUICE_CENTER_POS)
        
    def attach_glass_to_robot(self):
        """Attach glass to robot end effector"""
        if self.constraint_id != -1:
            p.removeConstraint(self.constraint_id)
        
        if self.robot_controller.end_effector_index == -1:
            ee_pos, ee_orn = p.getBasePositionAndOrientation(self.robot_id)
        else:
            ee_pos = p.getLinkState(self.robot_id, self.robot_controller.end_effector_index)[0]
        
        glass_pos, glass_orn = p.getBasePositionAndOrientation(self.glass_id)
        
        # Change the offset to position the glass on top of the robot
        # For R2D2, we need to place it higher
        offset = [0, 0, 0.3]  # Changed from [0, 0, 0.15] to position glass on top
        
        self.constraint_id = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=self.robot_controller.end_effector_index,
            childBodyUniqueId=self.glass_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=offset,
            childFramePosition=[0, 0, 0],
            childFrameOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        
        self.robot_controller.set_carrying_juice(True)

    def state_idle(self, dt):
        """Idle state - check for new orders"""
        self.order_manager.load_orders()
        order = self.order_manager.get_next_order()
        if order:
            print(f"New order: Table {order['table']}, Juice: {order['juice']}")
            self.current_table_index = order["table"]
            self.next_state = "MOVING_TO_JUICE"
        
    def state_moving_to_juice(self, dt):
        """Moving to juice center state"""
        current_pos = self.robot_controller.get_end_effector_position()
        
        if time.time() - self.state_start_time < dt * 2:
            self.robot_controller.plan_path(current_pos, JUICE_CENTER_POS)
            print(f"Planning path from {current_pos} to {JUICE_CENTER_POS}")
        
        path_completed = self.robot_controller.follow_path(dt)
        
        distance_to_goal = np.linalg.norm(np.array(current_pos) - np.array(JUICE_CENTER_POS))
        print(f"Distance to juice center: {distance_to_goal:.3f}, Path completed: {path_completed}")
        
        if time.time() - self.state_start_time > 10.0 and distance_to_goal > 1.0:
            print("Warning: Robot stuck, re-planning path")
            self.robot_controller.plan_path(current_pos, JUICE_CENTER_POS)
            self.state_start_time = time.time()
        
        if path_completed or distance_to_goal < 1:
            print("Reached juice center")
            self.next_state = "PICKING_GLASS"
            
    def state_picking_glass(self, dt):
        """Picking up glass state"""
        if time.time() - self.state_start_time > 1.0:
            self.attach_glass_to_robot()
            self.next_state = "FILLING_GLASS"
            
    def state_moving_to_table(self, dt):
        """Moving to customer table state"""
        if time.time() - self.state_start_time < dt:
            current_pos = self.robot_controller.get_end_effector_position()
            table_pos = TABLE_POSITIONS[self.current_table_index]
            # Table size is 0.6x0.6m; approach from the side (e.g., +x direction)
            table_size = [0.6, 0.6, 0.02]  # From create_table
            offset = [table_size[0]/2 + 0.2, 0, 0]  # 0.2m safety distance from edge
            delivery_pos = [
                table_pos[0] + offset[0],
                table_pos[1] + offset[1],
                table_pos[2] + 0.2  # Above table
            ]
            self.robot_controller.plan_path(current_pos, delivery_pos)
            print(f"Planning path to table {self.current_table_index} side: {delivery_pos}")
        
        path_completed = self.robot_controller.follow_path(dt)
        if path_completed:
            self.next_state = "LOWERING_GLASS"
            
    def state_lowering_glass(self, dt):
        """Lowering glass onto table state"""
        if time.time() - self.state_start_time < dt:
            current_pos = self.robot_controller.get_end_effector_position()
            table_pos = TABLE_POSITIONS[self.current_table_index]
            
            # Calculate a position that's reachable by the robot but places the glass on the table
            # Robot shouldn't move after approaching the table, just lower the glass
            
            # Get the robot's current XY position
            robot_xy = current_pos[:2]
            table_xy = table_pos[:2]
            
            # Direction vector from robot to table center
            direction = np.array(table_xy) - np.array(robot_xy)
            
            if np.linalg.norm(direction) > 0.01:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.array([1.0, 0.0])  # Default direction
                
            # Calculate a position on the table that's in front of the robot
            # The glass position should be between the robot and table center
            glass_xy = table_xy - direction * 0.2  # Move 20cm from table center toward robot
            
            delivery_pos = [
                glass_xy[0],
                glass_xy[1],
                table_pos[2] + 0.02  # Just above table surface
            ]
            
            self.robot_controller.plan_path(current_pos, delivery_pos)
            print(f"Planning path to place glass on table {self.current_table_index}: {delivery_pos}")
        
        path_completed = self.robot_controller.follow_path(dt)
        if path_completed:
            self.next_state = "DELIVERING_JUICE"
            
    def state_delivering_juice(self, dt):
        """Delivering juice state"""
        if time.time() - self.state_start_time > 0.5:
            # Get the table position
            table_pos = TABLE_POSITIONS[self.current_table_index]
            
            # Get current glass position before detaching
            glass_pos, glass_orn = p.getBasePositionAndOrientation(self.glass_id)
            
            # Detach glass from robot
            self.detach_glass_from_robot()
            
            # Explicitly place the glass on the table
            new_glass_pos = [
                table_pos[0],  # Center X of table
                table_pos[1],  # Center Y of table
                table_pos[2] + 0.1  # Height above table
            ]
            
            # Reset the glass position explicitly
            p.resetBasePositionAndOrientation(
                self.glass_id,
                new_glass_pos,
                p.getQuaternionFromEuler([0, 0, 0])
            )
            
            self.order_manager.complete_current_order()
            self.next_state = "MOVING_TO_HOME"
            
    def state_returning_home(self, dt):
        """Return home with proper low height path"""
        current_pos = self.robot_controller.get_end_effector_position()
        home_pos = HOME_POSITION.copy()
        home_pos[2] = 0
        
        # Plan return path if not already moving
        if time.time() - self.state_start_time < dt * 2:
            self.robot_controller.plan_path(current_pos, home_pos)
            print(f"Planning return path from {current_pos} to {home_pos}")
        
        # Follow return path
        path_completed = self.robot_controller.follow_path(dt)
        
        # Calculate distance to home position
        distance_to_home = np.linalg.norm(np.array(current_pos) - np.array(home_pos))
        print(f"Distance to home: {distance_to_home:.3f}, Path completed: {path_completed}")
        
        # If we're close enough to home or the path is complete
        if path_completed or distance_to_home < 0.5:
            print("Reached home position, resetting glass and transitioning to IDLE")
            self.reset_glass()
            self.next_state = "IDLE"
            
        # Failsafe in case robot gets stuck - replan after timeout
        if time.time() - self.state_start_time > 10.0 and distance_to_home > 0.5:
            print("Warning: Robot stuck returning home, re-planning path")
            self.robot_controller.plan_path(current_pos, home_pos)
            self.state_start_time = time.time()  # Reset timer for the new attempt

def create_glass(position):
    """Create a glass object with solid juice representation"""
    glass_height = 0.15
    glass_radius = 0.045
    visual_id = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=glass_radius,
        length=glass_height,
        rgbaColor=[0.9, 0.9, 0.95, 0.6]
    )
    collision_id = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=glass_radius,
        height=glass_height
    )
    glass_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=position,
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
    )
    
    juice_height = 0.0
    juice_radius = glass_radius * 0.9
    juice_visual_id = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=juice_radius,
        length=juice_height,
        rgbaColor=[1.0, 0.5, 0.0, 0.8]
    )
    juice_collision_id = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=juice_radius,
        height=juice_height
    )
    juice_id = p.createMultiBody(
        baseMass=0.01,
        baseCollisionShapeIndex=juice_collision_id,
        baseVisualShapeIndex=juice_visual_id,
        basePosition=[position[0], position[1], position[2] - glass_height/2 + juice_height/2],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
    )
    
    return glass_id, -1, {'juice_id': juice_id, 'juice_height': juice_height, 'glass_height': glass_height, 'juice_radius': juice_radius}

def create_table(position, size=[0.6, 0.6, 0.02], color=[0.6, 0.4, 0.2, 1.0]):
    """Create a table object"""
    half_extents = [size[0]/2, size[1]/2, size[2]/2]
    visual_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color
    )
    collision_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents
    )
    table_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=position
    )
    
    leg_radius = 0.03
    leg_height = position[2] * 2
    leg_positions = [
        [position[0] + half_extents[0] - leg_radius, position[1] + half_extents[1] - leg_radius, position[2] - leg_height/2],
        [position[0] + half_extents[0] - leg_radius, position[1] - half_extents[1] + leg_radius, position[2] - leg_height/2],
        [position[0] - half_extents[0] + leg_radius, position[1] + half_extents[1] - leg_radius, position[2] - leg_height/2],
        [position[0] - half_extents[0] + leg_radius, position[1] - half_extents[1] + leg_radius, position[2] - leg_height/2]
    ]
    
    leg_visual_id = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=leg_radius,
        length=leg_height,
        rgbaColor=color
    )
    leg_collision_id = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=leg_radius,
        height=leg_height
    )
    
    for leg_pos in leg_positions:
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=leg_collision_id,
            baseVisualShapeIndex=leg_visual_id,
            basePosition=leg_pos,
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
    
    return table_id


def create_obstacle(position, radius, height=0.3, color=[0.3, 0.3, 0.3, 1.0]):
    """Create cylindrical obstacle"""
    visual_id = p.createVisualShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color
    )
    collision_id = p.createCollisionShape(
        shapeType=p.GEOM_CYLINDER,
        radius=radius,
        height=height
    )
    obstacle_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=position
    )
    
    p.addUserDebugLine(
        [position[0] - radius, position[1], position[2]],
        [position[0] + radius, position[1], position[2]],
        [1, 0, 0],
        2.0
    )
    p.addUserDebugLine(
        [position[0], position[1] - radius, position[2]],
        [position[0], position[1] + radius, position[2]],
        [1, 0, 0],
        2.0
    )
    
    return obstacle_id

def create_robot():
    """Create and configure R2D2 robot"""
    robot_id = p.loadURDF("r2_server.urdf", basePosition=[0, 0, 0.5], globalScaling=1.0)
    
    p.changeDynamics(
        robot_id, 
        -1,
        linearDamping=0.1,
        angularDamping=0.1,
        maxJointVelocity=10.0
    )
    
    for i in range(p.getNumJoints(robot_id)):
        joint_info = p.getJointInfo(robot_id, i)
        joint_type = joint_info[2]
        
        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            p.resetJointState(robot_id, i, 0)
            p.changeDynamics(
                robot_id,
                i,
                linearDamping=0.1,
                angularDamping=0.1,
                jointLowerLimit=-3.14 if joint_type == p.JOINT_REVOLUTE else -0.5,
                jointUpperLimit=3.14 if joint_type == p.JOINT_REVOLUTE else 0.5,
            )
    
    return robot_id

def setup_environment():
    """Initialize PyBullet environment and create scene objects"""
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    plane_id = p.loadURDF("plane.urdf")
    
    # Initialize obstacles list with static obstacles
    obstacles = OBSTACLES.copy()
    
    # Create tables and add to obstacles
    tables = []
    for pos in TABLE_POSITIONS[1:]:  # Skip placeholder
        table_id = create_table(pos)
        tables.append(table_id)
        # Approximate table as a cylinder: use max dimension as radius
        table_size = [0.6, 0.6, 0.02]  # From create_table default
        table_radius = max(table_size[0], table_size[1]) / 2 + 0.1  # Add safety margin
        obstacles.append([pos[0], pos[1], pos[2], table_radius])
    
    # Create juice counter (bar) and add to obstacles
    bar_pos = [JUICE_CENTER_POS[0], JUICE_CENTER_POS[1], JUICE_CENTER_POS[2] - 0.1]
    bar_size = [1.0, 0.8, 0.04]  # From create_table call
    bar_id = create_table(bar_pos, size=bar_size, color=[0.8, 0.8, 0.8, 1.0])
    bar_radius = max(bar_size[0], bar_size[1]) / 2 + 0.1  # Add safety margin
    obstacles.append([bar_pos[0], bar_pos[1], bar_pos[2], bar_radius])
    
    # Create static obstacles
    for obs in OBSTACLES:
        obstacle_id = create_obstacle([obs[0], obs[1], obs[2]], obs[3])
    
    robot_id = create_robot()
    
    glass_id, _, _ = create_glass(JUICE_CENTER_POS)
    
    p.resetDebugVisualizerCamera(
        cameraDistance=3.0,
        cameraYaw=50,
        cameraPitch=-35,
        cameraTargetPosition=[0, 0, 0.5]
    )
    
    return robot_id, glass_id, obstacles


def initialize_data_file():
    """Create initial data.json if not exists"""
    if not os.path.exists("data.json"):
        default_data = {
            "orders": [
                {"table": 2, "juice": "orange"},
                {"table": 2, "juice": "orange"},
                {"table": 3, "juice": "orange"}
            ]
        }
        with open("data.json", "w") as f:
            json.dump(default_data, f)

def monitor_keyboard_input(state_machine):
    """Monitor keyboard input for testing"""
    while True:
        key = input("Enter command ('q' to quit, 'n' for new order): ")
        if key == 'q':
            break
        elif key == 'n':
            with open("data.json", "r") as f:
                data = json.load(f)
            
            data["orders"].append({
                "table": np.random.randint(1, len(TABLE_POSITIONS)),
                "juice": "orange"
            })
            
            with open("data.json", "w") as f:
                json.dump(data, f)
            
            print("New order added")


def main():
    """Main program loop"""
    initialize_data_file()
    
    # Setup environment and get obstacles
    robot_id, glass_id, obstacles = setup_environment()
    
    # Update global OBSTACLES to include tables and juice counter
    global OBSTACLES
    OBSTACLES = obstacles
    
    state_machine = RobotBartenderStateMachine(robot_id, glass_id)
    
    input_thread = threading.Thread(target=monitor_keyboard_input, args=(state_machine,))
    input_thread.daemon = True
    input_thread.start()
    
    step_size = 1/240.0
    update_interval = 1/60.0
    time_accumulator = 0.0
    last_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            time_accumulator += dt
            while time_accumulator >= update_interval:
                state_machine.update(update_interval)
                time_accumulator -= update_interval
            
            p.stepSimulation()
            
            time.sleep(max(0, step_size - (time.time() - current_time)))
    
    except KeyboardInterrupt:
        print("Simulation terminated by user")
    finally:
        p.disconnect()


# Define global constants
HOME_POSITION = [0.3, 0.0, 0.6]
JUICE_CENTER_POS = [0.0, 0.9, 0.2]
TABLE_POSITIONS = [
    [0, 0, 0],  # Placeholder for indexing
    [1.2, 0.8, 0.2],  # Table 1
    [1.2, -0.8, 0.2],  # Table 2
    [-1.2, 0.8, 0.2],  # Table 3
    [-1.2, -0.8, 0.2]   # Table 4
]
OBSTACLES = [
    [0.5, 0.3, 0.3, 0.15],  # [x, y, z, radius]
    [-0.5, -0.3, 0.3, 0.15],
    [0.0, -0.5, 0.3, 0.12]
]

if __name__ == "__main__":
    main()