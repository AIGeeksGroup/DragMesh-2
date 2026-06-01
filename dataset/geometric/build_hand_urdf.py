"""
Generate an augmented SMPLX right hand URDF with 3 prismatic virtual joints
prepended for free 3D positioning.

DOF layout of the output URDF:
  0: virtual_x  (prismatic, X-axis)
  1: virtual_y  (prismatic, Y-axis)
  2: virtual_z  (prismatic, Z-axis)
  3-5: R_Wrist_x/y/z  (revolute, wrist rotation)
  6-50: finger joints  (45 DOFs)
  Total: 51 DOFs

Usage:
    python dataset/geometric/build_hand_urdf.py \
        --input  hands/smplx_variants/hands_only/smplx_right_hand.urdf \
        --output hands/floating/smplx_right_hand_floating.urdf
"""

import argparse
import xml.etree.ElementTree as ET


VIRTUAL_JOINTS_XML = """\
  <!-- ========== Virtual prismatic joints for free positioning ========== -->
  <link name="floating_base">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1e-06"/>
      <inertia ixx="1e-09" ixy="0" ixz="0" iyy="1e-09" iyz="0" izz="1e-09"/>
    </inertial>
  </link>

  <link name="slide_x_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1e-06"/>
      <inertia ixx="1e-09" ixy="0" ixz="0" iyy="1e-09" iyz="0" izz="1e-09"/>
    </inertial>
  </link>
  <joint name="virtual_x" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="floating_base"/>
    <child  link="slide_x_link"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.0" upper="2.0" effort="1000" velocity="10"/>
    <dynamics damping="50" friction="0"/>
  </joint>

  <link name="slide_y_link">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="1e-06"/>
      <inertia ixx="1e-09" ixy="0" ixz="0" iyy="1e-09" iyz="0" izz="1e-09"/>
    </inertial>
  </link>
  <joint name="virtual_y" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="slide_x_link"/>
    <child  link="slide_y_link"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="1000" velocity="10"/>
    <dynamics damping="50" friction="0"/>
  </joint>

  <joint name="virtual_z" type="prismatic">
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <parent link="slide_y_link"/>
    <child  link="R_Wrist_Base"/>
    <axis xyz="0 0 1"/>
    <limit lower="0.0" upper="2.0" effort="1000" velocity="10"/>
    <dynamics damping="50" friction="0"/>
  </joint>
"""


def build_floating_hand_urdf(input_path: str, output_path: str):
    tree = ET.parse(input_path)
    root = tree.getroot()
    root.set("name", "smplx_right_hand_floating")

    # Parse virtual joints XML and insert at the beginning
    virtual_elements = ET.fromstring(f"<dummy>{VIRTUAL_JOINTS_XML}</dummy>")
    # Insert in reverse so they end up in order at position 0
    for elem in reversed(list(virtual_elements)):
        root.insert(0, elem)

    tree.write(output_path, encoding="unicode", xml_declaration=True)
    print(f"Wrote augmented URDF to {output_path}")
    print(f"  Total DOFs: 3 prismatic + 48 revolute = 51")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="hands/smplx_variants/hands_only/smplx_right_hand.urdf")
    parser.add_argument("--output", default="hands/floating/smplx_right_hand_floating.urdf")
    args = parser.parse_args()
    build_floating_hand_urdf(args.input, args.output)
