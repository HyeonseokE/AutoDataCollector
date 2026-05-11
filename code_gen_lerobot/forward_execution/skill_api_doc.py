"""
Robot API Skills Documentation (Gemini Robotics class definition style)

LeRobotSkills 클래스의 API 문서를 LLM 프롬프트용으로 제공합니다.
실제 구현은 skills/skills_lerobot.py에 있으며,
이 파일은 LLM이 코드 생성 시 참조할 API 명세만 포함합니다.
"""

ROBOT_API_DOC = '''class LeRobotSkills:
    """Interface for controlling the LeRobot SO-101 single robot arm.
    The robot has a 5-DOF arm (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll)
    with an asymmetric two-finger gripper (left finger is fixed, right finger is actuated).
    The gripper can open up to 0.07m (7cm) and approaches objects from directly above (top-down grasp).
    All positions are specified in the world coordinate frame in meters.

    IMPORTANT: Every skill method accepts two optional string parameters for dataset recording:
        skill_description (str): Concise sentence describing the action and purpose.
            Example: "Move gripper above chocolate_pie_1 to prepare for picking"
        verification_question (str): Yes/No question to visually verify the action's outcome.
            Example: "Is the gripper positioned above chocolate_pie_1?"
    You MUST always pass both parameters for every skill call.
    """

    def connect(self) -> bool:
        """Connects to robot hardware and initializes kinematics. Returns True if successful."""

    def disconnect(self):
        """Disconnects from robot hardware. Must be called in a finally block for cleanup."""

    def gripper_open(self, duration: float = 1.5, ratio: float = 1.0):
        """Opens the gripper to the specified ratio. STANDALONE call (arm stays still).

        Prefer integrated gripper motion via `move_to_position(..., gripper_action="open")`
        for pick approaches. Use this standalone variant only when the arm must not move.

        Args:
            duration: Movement duration in seconds.
            ratio: Open ratio where 0.0 = fully closed and 1.0 = fully open.
                Use 0.7 for partial open during place operations for controlled release.
        """

    def gripper_close(self, duration: float = 1.5):
        """Closes the gripper to grasp an object. STANDALONE call (arm stays still).

        Prefer integrated gripper motion via `move_to_position(..., gripper_action="close")`
        for place retreats. Use this standalone variant only when the arm must not move
        (e.g., before execute_push / execute_press where arm is already in position).
        The gripper closes with sufficient force to hold objects up to ~500g.

        Args:
            duration: Movement duration in seconds.
        """

    def move_to_initial_state(self) -> bool:
        """Moves the arm to its home position. Call at the start of every task."""

    def move_to_free_state(self) -> bool:
        """Moves the arm to a safe parking position. Call as the very last skill after task completion."""

    def move_to_position(self, position: list[float], duration: float = None,
                         target_name: str = None,
                         gripper_action: str = None,
                         gripper_start_fraction: float = 0.0,
                         gripper_end_fraction: float = 1.0,
                         gripper_open_ratio: float = 1.0) -> bool:
        """Moves the end-effector to the given XYZ position in world coordinates.
        Use this for approach movements (moving above an object before pick/place),
        retreat movements (lifting after pick/place), and transit movements between objects.

        The world coordinate frame origin is at the rear-center of the workspace table
        on the table surface:
            Positive x: towards front of the table
            Negative x: towards back of the table
            Positive y: towards right
            Negative y: towards left
            Positive z: up, towards ceiling (z=0 is table surface)

        INTEGRATED GRIPPER MOTION:
            Use `gripper_action` to transition the gripper concurrently with the arm motion.
            This produces ONE skill event (one dataset label) that captures both arm + gripper.

            - gripper_action=None (default): gripper stays at current position (legacy behavior).
            - gripper_action="open":  gripper opens during the motion.
                For pick approach, use: gripper_action="open", gripper_start_fraction=0.3
                (gripper opens during the last 70% of the approach).
            - gripper_action="close": gripper closes during the motion.
                For place retreat, use: gripper_action="close", gripper_start_fraction=0.7
                (gripper closes during the last 30% of the retreat).

            When gripper_action is used, write skill_description as a compound sentence:
                "Approach <obj> and open gripper"
                "Retreat from <target> and close gripper"

        Args:
            position: Target position [x, y, z] in meters in world frame.
            duration: Movement duration in seconds. Uses default if None.
            target_name: Name of the target object for subgoal labeling in dataset recording.
                Example: "yellow dice", "blue dish".
            gripper_action: Optional "open" or "close" for concurrent gripper motion. None = hold.
            gripper_start_fraction: Fraction of motion duration at which gripper interpolation begins (0.0–1.0).
            gripper_end_fraction:   Fraction of motion duration at which gripper interpolation ends (0.0–1.0).
            gripper_open_ratio: Target open ratio when gripper_action="open" (default 1.0 = fully open).

        Returns:
            True if movement successful, False if position is outside reachable workspace.
        """

    def rotate_90degree(self, direction: int = 1, duration: float = 2.0) -> bool:
        """Rotates the gripper (wrist_roll joint) by 90 degrees in place.
        The arm position remains the same; only the gripper orientation changes.
        Use this when an object needs to be reoriented after picking.

        Args:
            direction: 1 for clockwise rotation, -1 for counter-clockwise rotation.
            duration: Movement duration in seconds.

        Returns:
            True if rotation successful.
        """

    def execute_pick_object(self, object_position: list[float],
                            object_name: str = None) -> bool:
        """Executes a pick (grasp) action at the given object position.
        Must be called AFTER moving to the approach position above the object.
        The robot descends to the grasp height (2.5cm offset from object top),
        closes the gripper to grasp the object, and internally saves the current
        pitch angle for the subsequent place operation.

        IMPORTANT: Pass the object position as-is from the positions dictionary.
        The function internally calculates the grasp height (2.5cm below object top).
        Do NOT subtract any offset from z — just pass pick_pos directly.

        Args:
            object_position: Object position [x, y, z] in meters. Pass as-is from
                positions dictionary. The function internally handles the grasp offset.
            object_name: Name of the object being picked for subgoal labeling.
                Example: "yellow dice", "red cup".

        Returns:
            True if pick successful (gripper closed around object).
        """

    def execute_place_object(self, place_position: list[float],
                             is_table: bool = True,
                             gripper_open_ratio: float = 1.0, target_name: str = None) -> bool:
        """Executes a place (release) action at the given target position.
        Must be called AFTER moving to the approach position above the target.
        The robot descends to the place height with the pitch angle saved during
        the pick operation automatically restored, then opens the gripper to
        release the object.

        IMPORTANT: Pass the target surface position as-is from the positions dictionary.
        The function internally calculates the correct release height using the pick height saved during execute_pick_object.
        - is_table=True: z value is ignored (release height = pick_z above table surface)
        - is_table=False: z value is used as the target surface height (release height = surface_z + pick_z)
        ALWAYS use gripper_open_ratio=0.7 for controlled release.

        Args:
            place_position: Target position [x, y, z] in meters. Pass the target object/surface
                position as-is. The z coordinate is only used when is_table=False.
            is_table: True if placing directly on the table surface (z=0),
                False if placing on top of another object.
            gripper_open_ratio: How much to open the gripper for release (0.0 to 1.0).
                ALWAYS use 0.7 (70% open) for controlled object release.
            target_name: Name of the placement target for subgoal labeling.
                Example: "blue dish", "table".

        Returns:
            True if place successful (object released at target position).
        """

    def execute_place_lid(self, place_position: list[float],
                          pull_distance: float = 0.02,
                          gripper_open_ratio: float = 0.7,
                          target_name: str = None) -> bool:
        """Executes a lid-specific place: descends onto the container, drags the lid
        in the -x direction by `pull_distance`, then releases.

        USE THIS — NOT execute_place_object — whenever the held object is a lid
        (e.g. pot lid) being seated on top of a container. Lids systematically land
        slightly farther from the robot (+x) than the true container center; the
        drag step pulls the lid back toward the robot before release so the lid
        sits centered on the rim.

        Must be called AFTER moving to the approach position above the container
        with the lid already grasped (saved pitch from execute_pick_object is
        automatically restored). The release height is computed exactly like
        execute_place_object(is_table=False): surface_z + saved pick_z.

        Args:
            place_position: Container top-surface position [x, y, z] in meters.
                Pass the container's position as-is from the positions dictionary
                (e.g., positions["pot"]["position"]). z is the surface height.
            pull_distance: -x drag distance in meters before release (default 0.02 = 2cm).
                Leave at default unless explicitly instructed otherwise.
            gripper_open_ratio: How much to open the gripper for release (default 0.7).
                ALWAYS use 0.7 (70% open) for controlled release.
            target_name: Container label for subgoal recording (e.g. "pot").

        Returns:
            True if descent, drag, and release all succeed.
        """

    def execute_press(self, position: list[float], press_depth: float = 0.01,
                      contact_height: float = 0.02, press_duration: float = 0.5,
                      hold_time: float = 0.3, max_press_torque: int = 400,
                      duration: float = None,
                      target_name: str = None) -> bool:
        """Executes a 2-phase press action (normal descent + torque-limited press).
        Must be called AFTER closing the gripper and moving to approach position above target.
        Phase 1: Descend to contact surface at normal speed.
        Phase 2: Press below contact with torque limit for safe force application.

        Args:
            position: Target position [x, y, z] in meters (center of press target).
            press_depth: How far to press below the contact surface in meters (default 0.01 = 1cm).
            contact_height: Height of the contact surface in meters (use object's z value).
            press_duration: Duration for the pressing phase in seconds.
            hold_time: How long to hold at pressed position in seconds (default 0.3).
            max_press_torque: Torque limit during press phase (0-1000, default 400).
            duration: Duration for the descent phase. Uses default if None.
            target_name: Name of the target for subgoal labeling.
                Example: "power button", "microphone".

        Returns:
            True if press action completed successfully.
        """

    def execute_push(self, start_position: list[float], distance: float,
                     duration: float = None, object_name: str = None) -> bool:
        """CLOSE a drawer/door — push handle by `distance + 3cm` in +x.

        Pre: caller approached above start at approach_height with gripper OPEN
        (via `gripper_action="open"`). The open jaws act as a paddle pushing the
        handle from inside.

        The caller passes the **same distance the drawer was opened by** (or extracts
        from the user instruction). The skill internally pushes `distance + 3cm` so
        the drawer is fully closed (overshoot margin). Direction is fixed at +x.

        Internally: descend (over-descent + sag bypass for handle z) → linear push
        +x by (distance + 3cm) at handle z (pitch locked) → retreat-with-close.
        Motor torque is limited during the push (compliance — protects motor and
        drawer when the close stop is reached).

        Args:
            start_position: Current handle position (in open state) [x, y, z] meters.
            distance: Nominal close distance (meters); typically equals the opening
                distance. The skill internally pushes distance + 0.03m. Example: if
                forward opened with 10cm → close with distance=0.10 (skill pushes 12cm).
            duration: Push movement duration (seconds, None = auto from distance).
            object_name: Object label for subgoal recording.

        Returns:
            True if push completed.
        """

    def execute_pull(self, start_position: list[float], distance: float,
                     duration: float = None, object_name: str = None) -> bool:
        """OPEN a drawer/door — pull handle by `distance` meters in -x direction.

        Pre: caller approached above start at approach_height with gripper OPEN
        (via `gripper_action="open"`).

        The caller passes `distance` (typically extracted from the user instruction,
        e.g., "Open the drawer 10cm" → 0.10). Direction is fixed at -x (toward
        robot base).

        Internally: descend → close gripper (grasp) → linear pull -x by `distance`
        at handle z (pitch locked) → open gripper (release) → retreat-with-close.

        Args:
            start_position: Handle grasp point [x, y, z] meters.
            distance: Pull distance (meters, e.g., 0.10 = 10cm). Positive value;
                direction is fixed at -x.
            duration: Pull movement duration (seconds, None = auto from distance).
            object_name: Object label for subgoal recording.

        Returns:
            True if pull completed successfully.
        """

    def detect_objects(self, queries: list[str], timeout: float = 5.0,
                       visualize: bool = False) -> dict:
        """Re-detects objects in real-time using the camera during code execution.

        SIDE EFFECT: Automatically merges fresh detections into the global
        ``positions`` dict in-place AND preserves Turn 2 point labels
        (e.g. "top placement point", "grasp center"). Objects whose detection
        fails simply keep their previous values — no fallback boilerplate.

        Args:
            queries: List of object names to detect. Example: ["brown block", "green block"].
            timeout: Detection timeout in seconds (default 5.0).
            visualize: Whether to show detection visualization window (default False).

        Returns:
            Dict mapping object name to position info (same shape as initial detection).
            Returns None for objects that could not be detected. The return
            value mirrors what was merged into ``positions``; in typical
            generated code you read from ``positions`` directly.

        Example:
            # After placing block A, re-detect — `positions` is updated in-place.
            skills.detect_objects(["block A"])
            new_pos = positions["block A"]["position"]
            skills.execute_place_object(new_pos, is_table=False, ...)
        """

    def set_subtask(self, description: str) -> None:
        """Set sub-task label for recording. Groups multiple skills under one higher-level label.
        Call this before each object's pick-place sequence.

        Args:
            description: Short description of the subtask (e.g., "pick red block and place at center")
        """

    def clear_subtask(self) -> None:
        """Clear the current sub-task label. Call after all objects are moved."""

    def detect_objects(self, queries: list) -> dict:
        """Re-detect objects in the current camera view using VLM + depth sensor.

        SIDE EFFECT: Automatically merges fresh detections into the global
        ``positions`` dict in-place AND preserves Turn 2 point labels
        (e.g. "top placement point", "grasp center"). Objects whose detection
        fails simply keep their previous values.

        Returns updated positions including z height (from RealSense depth).
        IMPORTANT: Always call this after set_subtask() to refresh object positions
        before picking/placing. This is critical when objects have been moved
        (e.g., stacked on top of each other, changing their z height).

        Args:
            queries: List of object names to detect, e.g., ["red block", "yellow block"]

        Returns:
            Dict: {"object_name": {"position": [x, y, z], "pixel": (u, v), "bbox_px": (w, h)}}
            (Mirrors what was merged into ``positions``; usually read from
            ``positions`` directly.)

        Pattern — always follow this sequence when switching to a new object:
            # 1. Set subtask label
            skills.set_subtask("pick purple block and place at target")
            # 2. Move to initial state (clear arm from camera view before detection)
            skills.move_to_initial_state()
            # 3. Re-detect — `positions` is updated in-place, point labels preserved.
            skills.detect_objects(["red block", "yellow block", "purple block"])
            # 4. Re-extract local variables from the refreshed `positions`.
            purple_pos = positions["purple block"]["position"]
            target_pos = positions["yellow block"]["position"]
            # 5. Approach with integrated gripper open, then pick
            skills.move_to_position([purple_pos[0], purple_pos[1], 0.20],
                                    target_name="purple block",
                                    gripper_action="open", gripper_start_fraction=0.3)
            skills.execute_pick_object(purple_pos, object_name="purple block")
        """'''
