"""
Multi-Arm Robot API Skills Documentation

LLM이 코드 생성 시 참조할 API 명세.
Pipeline이 `skills = MultiArmSkills(...)` 를 사전 생성하여 exec_globals에 주입.
"""

MULTI_ARM_API_DOC = '''**Available Robot API** (the `skills` object is already created — do NOT import or instantiate it):

    `skills` is a MultiArmSkills instance controlling two SO-101 robot arms.
    Each arm has a 5-DOF arm + gripper.
    All methods use left_arm / right_arm parameters. Pass "wait" to skip one arm (it holds position).

    IMPORTANT: Every skill method accepts per-arm string parameters:
        left_skill_description / right_skill_description (str): Concise sentence describing each arm's action.
        left_verification_question / right_verification_question (str): Yes/No question to verify each arm's outcome.
    You MUST always pass both parameters for every arm that is NOT "wait".

    def connect(self) -> bool:
        """Connects both robot arms and initializes kinematics. Returns True if both succeed."""

    def disconnect(self):
        """Disconnects both robot arms. Must be called in a finally block for cleanup."""

    def move_to_position(self, left_arm="wait", right_arm="wait",
                         left_duration=None, right_duration=None,
                         left_skill_description=None, right_skill_description=None,
                         left_verification_question=None, right_verification_question=None) -> dict:
        """Move arms to target positions. Pass "wait" to skip one arm.

        Args:
            left_arm: [x,y,z] for left arm, or "wait" to hold position.
            right_arm: [x,y,z] for right arm, or "wait" to hold position.
        """

    def pick_object(self, left_arm="wait", right_arm="wait",
                    left_object_name=None, right_object_name=None,
                    left_skill_description=None, right_skill_description=None,
                    left_verification_question=None, right_verification_question=None) -> dict:
        """Pick objects. Pass "wait" to skip one arm.

        Args:
            left_arm: Object position [x,y,z] for left arm, or "wait".
            right_arm: Object position [x,y,z] for right arm, or "wait".
        """

    def place_object(self, left_arm="wait", right_arm="wait",
                     left_is_table=True, right_is_table=True,
                     left_skill_description=None, right_skill_description=None,
                     left_verification_question=None, right_verification_question=None) -> dict:
        """Place objects. Pass "wait" to skip one arm.

        Args:
            left_arm: Place position [x,y,z] for left arm, or "wait".
            right_arm: Place position [x,y,z] for right arm, or "wait".
        """

    def move_to_pixel(self, left_arm="wait", right_arm="wait",
                      left_skill_description=None, right_skill_description=None,
                      left_verification_question=None, right_verification_question=None) -> dict:
        """Move arms to positions specified by normalized pixel coordinates.
        Pass "wait" to skip one arm.

        Args:
            left_arm: [y, x] in 0-1000 range for left arm, or "wait".
            right_arm: [y, x] in 0-1000 range for right arm, or "wait".
        """

    def place_at_pixel(self, left_arm="wait", right_arm="wait",
                       left_is_table=True, right_is_table=True,
                       left_skill_description=None, right_skill_description=None,
                       left_verification_question=None, right_verification_question=None) -> dict:
        """Place at positions specified by normalized pixel coordinates.
        Pass "wait" to skip one arm.

        Args:
            left_arm: [y, x] in 0-1000 range for left arm, or "wait".
            right_arm: [y, x] in 0-1000 range for right arm, or "wait".
        """

    def gripper_control(self, left_arm="wait", right_arm="wait",
                        left_duration=1.5, right_duration=1.5,
                        left_ratio=1.0, right_ratio=1.0,
                        left_skill_description=None, right_skill_description=None,
                        left_verification_question=None, right_verification_question=None) -> dict:
        """Control grippers.

        Args:
            left_arm: "open", "close", or "wait".
            right_arm: "open", "close", or "wait".
        """

    def move_to_initial_state(self) -> dict:
        """Move both arms to their home positions simultaneously."""

    def move_to_free_state(self) -> dict:
        """Move both arms to safe parking positions simultaneously."""

    def set_subtask(self, description: str) -> None:
        """Set sub-task label for recording. Call before each subtask begins.
        Args:
            description: Natural language description of what this subtask does.
                         e.g., "pick red block with left arm and place at center"
        """

    def clear_subtask(self) -> None:
        """Clear sub-task label. Call after each subtask ends."""

    def detect_objects(self, queries: list[str], timeout: float = 5.0) -> dict:
        """Re-detect objects and return updated positions for both arms.
        Returns: {"left_arm": {obj: {"position": [...], ...}}, "right_arm": {...}}
        """
'''
