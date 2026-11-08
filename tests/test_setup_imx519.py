import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import setup_imx519 as setup


class PlatformTests(unittest.TestCase):
    def test_os_release_is_read_without_executing_shell_content(self):
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "os-release"
            release.write_text('PRETTY_NAME="Debian GNU/Linux 12 (bookworm)"\nVERSION_CODENAME=bookworm\n')
            values = setup.read_os_release(release)
            self.assertEqual(values["VERSION_CODENAME"], "bookworm")
            self.assertEqual(values["PRETTY_NAME"], "Debian GNU/Linux 12 (bookworm)")

    def test_pi4_os_and_kernel_requirements_are_checked(self):
        for codename, kernel, supported in (("bookworm", "6.6.51+rpt-rpi-v8", True),
                                            ("trixie", "6.12.47+rpt-rpi-v8", True),
                                            ("bullseye", "6.1.21-v8+", True),
                                            ("bullseye", "5.15.84-v8+", False),
                                            ("buster", "6.1.21-v8+", False)):
            with self.subTest(codename=codename, kernel=kernel), \
                    patch.object(setup.sys, "platform", "linux"), \
                    patch.object(Path, "read_text", return_value="Raspberry Pi 4 Model B Rev 1.4\x00"), \
                    patch.object(setup, "read_os_release", return_value={"VERSION_CODENAME": codename}), \
                    patch.object(setup.platform, "release", return_value=kernel), \
                    contextlib.redirect_stdout(io.StringIO()):
                if supported:
                    self.assertEqual(setup.check_platform(), codename)
                else:
                    with self.assertRaises(setup.SetupError):
                        setup.check_platform()


class BootConfigTests(unittest.TestCase):
    def test_old_cameras_are_disabled_without_changing_display_or_gpio_settings(self):
        original = (
            "# Existing Pi configuration\n[all]\ncamera_auto_detect=1\n"
            "dtoverlay=imx219\ndtoverlay=vc4-kms-v3d,cma-256\n"
            "dtparam=audio=on\ndtoverlay=gpio-shutdown,gpio_pin=17\n"
            "[pi4]\ndtoverlay=arducam-pivariety\n[none]\n"
        )
        configured = setup.configure_text(original, root=True)
        active = [line for line in configured.splitlines() if not line.startswith("#")]
        self.assertEqual(active.count("dtoverlay=imx519"), 1)
        self.assertEqual(active.count("camera_auto_detect=0"), 1)
        self.assertNotIn("camera_auto_detect=1", active)
        self.assertNotIn("dtoverlay=imx219", active)
        self.assertNotIn("dtoverlay=arducam-pivariety", active)
        for unchanged in ("dtoverlay=vc4-kms-v3d,cma-256", "dtparam=audio=on",
                          "dtoverlay=gpio-shutdown,gpio_pin=17"):
            self.assertIn(unchanged, active)
        self.assertTrue(configured.endswith("[all]\n" + setup.BLOCK))
        self.assertEqual(setup.configure_text(configured, root=True), configured)

    def test_existing_sensor_parameters_and_multiplexer_overlays_are_replaced(self):
        original = (
            "[all]\n  dtoverlay = imx519,cam0,vcm=off # old setup\n"
            "dtoverlay=camera-mux-4port,cam0-imx219,cam1-imx219\n"
            "dtoverlay=ov5647\ncamera_auto_detect = 1\n"
        )
        configured = setup.configure_text(original, root=True)
        active = [line for line in configured.splitlines() if not line.startswith("#")]
        self.assertEqual([line for line in active if "dtoverlay" in line], ["dtoverlay=imx519"])
        self.assertEqual(setup.configure_text(configured, root=True), configured)

    def test_settings_added_after_a_managed_block_are_preserved_on_rerun(self):
        previous = "[all]\n" + setup.BLOCK + "\ndtparam=spi=on\n"
        configured = setup.configure_text(previous, root=True)
        self.assertIn("\ndtparam=spi=on\n", configured)
        self.assertEqual(configured.count(setup.BEGIN), 1)
        self.assertEqual(setup.configure_text(configured, root=True), configured)

    def test_unrelated_include_contents_are_unchanged(self):
        for original in ("dtoverlay=vc4-kms-v3d\n\n\n", "# Comment without newline", ""):
            self.assertEqual(setup.configure_text(original, root=False), original)

    def test_incomplete_managed_block_is_rejected(self):
        with self.assertRaisesRegex(setup.SetupError, "Malformed"):
            setup.configure_text(setup.BEGIN + "\ncamera_auto_detect=0\n", root=True)

    def test_recursive_includes_and_cycles_are_resolved_from_the_boot_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            boot = Path(directory)
            config = boot / "config.txt"
            config.write_text("[all]\ninclude custom/usercfg.txt\n")
            (boot / "custom").mkdir()
            child = boot / "custom" / "usercfg.txt"
            child.write_text("dtoverlay=imx219\ninclude config.txt\n")
            updates = setup.config_updates(config)
            self.assertEqual(set(updates), {config.resolve(), child.resolve()})
            self.assertNotIn("\ndtoverlay=imx219\n", "\n" + updates[child.resolve()][1])
            self.assertNotIn(setup.BLOCK, updates[child.resolve()][1])

    def test_includes_outside_the_boot_directory_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            boot = base / "boot"
            boot.mkdir()
            (base / "outside.txt").write_text("dtoverlay=imx219\n")
            config = boot / "config.txt"
            config.write_text("include ../outside.txt\n")
            with self.assertRaisesRegex(setup.SetupError, "outside"):
                setup.config_updates(config)

    def test_backups_preserve_original_bytes_and_config_replacement_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.txt"
            original = b"[all]\r\ncamera_auto_detect=1\r\ndtoverlay=imx219\r\n"
            config.write_bytes(original)
            with contextlib.redirect_stdout(io.StringIO()):
                setup.backup_configs([config])
            configured = setup.configure_text(config.read_text(), root=True)
            setup.write_config(config, configured)
            backups = list(config.parent.glob("*.bak"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            self.assertEqual(config.read_text(), configured)
            self.assertEqual(list(config.parent.glob(".wigglegram-*")), [])

    def test_boot_config_location_depends_on_os(self):
        self.assertEqual(setup.boot_config_path("bullseye"), Path("/boot/config.txt"))
        for codename in ("bookworm", "trixie"):
            self.assertEqual(setup.boot_config_path(codename), Path("/boot/firmware/config.txt"))


class SetupFlowTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.boot = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.config = self.boot / "config.txt"
        self.original = "[all]\ncamera_auto_detect=1\ndtoverlay=imx219\n"
        self.config.write_text(self.original)
        (self.boot / "overlays").mkdir()
        self.overlay = self.boot / "overlays" / "imx519.dtbo"
        self.overlay.write_bytes(b"test overlay")
        self.stack.enter_context(patch.object(setup, "check_platform", return_value="bookworm"))
        self.stack.enter_context(patch.object(setup, "boot_config_path", return_value=self.config))
        self.stack.enter_context(patch.object(setup.os, "geteuid", return_value=0, create=True))
        self.install = self.stack.enter_context(patch.object(setup, "install_camera_stack"))
        self.stdout = self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stderr = self.stack.enter_context(contextlib.redirect_stderr(io.StringIO()))

    def test_dry_run_makes_no_downloads_backups_or_config_changes(self):
        with patch.object(setup, "urlopen") as download:
            self.assertEqual(setup.main(["--dry-run"]), 0)
        self.install.assert_not_called()
        download.assert_not_called()
        self.assertEqual(self.config.read_text(), self.original)
        self.assertEqual(list(self.boot.glob("*.bak")), [])
        self.assertIn("+dtoverlay=imx519", self.stdout.getvalue())

    def test_install_backs_up_before_running_vendor_and_preserves_vendor_config_edits(self):
        def install():
            backups = list(self.boot.glob("*.bak"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), self.original)
            self.config.write_text(self.original + "dtparam=spi=on\n")

        self.install.side_effect = install
        self.assertEqual(setup.main([]), 0)
        self.install.assert_called_once()
        configured = self.config.read_text()
        self.assertIn("dtparam=spi=on\n", configured)
        self.assertTrue(configured.endswith(setup.BLOCK))
        self.assertIn("sudo reboot", self.stdout.getvalue())
        self.assertEqual(setup.main(["--config-only"]), 0)
        self.assertEqual(self.config.read_text(), configured)

    def test_config_only_skips_vendor_installation(self):
        self.assertEqual(setup.main(["--config-only"]), 0)
        self.install.assert_not_called()
        self.assertTrue(self.config.read_text().endswith(setup.BLOCK))

    def test_vendor_failure_keeps_original_config_and_reports_failure(self):
        self.install.side_effect = subprocess.CalledProcessError(1, ["bash", "vendor-installer"])
        self.assertEqual(setup.main([]), 1)
        self.assertEqual(self.config.read_text(), self.original)
        self.assertEqual(len(list(self.boot.glob("*.bak"))), 1)
        self.assertNotIn("Camera setup complete", self.stdout.getvalue())

    def test_missing_include_fails_before_installing_or_backing_up(self):
        self.config.write_text(self.original + "include missing.txt\n")
        self.assertEqual(setup.main([]), 1)
        self.install.assert_not_called()
        self.assertEqual(list(self.boot.glob("*.bak")), [])

    def test_missing_driver_overlay_does_not_claim_success_or_change_boot_config(self):
        self.overlay.unlink()
        self.assertEqual(setup.main(["--config-only"]), 1)
        self.assertEqual(self.config.read_text(), self.original)
        self.assertIn("imx519.dtbo", self.stderr.getvalue())

    def test_post_reboot_check_requires_an_enumerated_imx519_device(self):
        for output, expected in (("Available cameras\n0 : imx519 [4656x3496]\n", 0),
                                 ("No cameras available!\n", 1)):
            with self.subTest(output=output), \
                    patch.object(setup, "camera_executable", return_value="rpicam-still"), \
                    patch.object(setup.subprocess, "run", return_value=Mock(returncode=0, stdout=output)) as run:
                self.assertEqual(setup.main(["--check"]), expected)
                self.assertEqual(run.call_args.args[0], ["rpicam-still", "--list-cameras"])
        self.install.assert_not_called()
        self.assertEqual(self.config.read_text(), self.original)


class VendorInstallTests(unittest.TestCase):
    def test_exact_guide_packages_are_installed_in_order_and_commands_are_verified(self):
        with patch.object(setup, "run") as run, \
                patch.object(setup, "urlopen", return_value=io.BytesIO(b"#!/bin/bash\nexit 0\n")), \
                patch.object(setup.subprocess, "run", return_value=Mock(stdout="")), \
                patch.object(setup, "camera_executable", side_effect=lambda kind: f"rpicam-{kind}"), \
                contextlib.redirect_stdout(io.StringIO()):
            setup.install_camera_stack()
        commands = [call.args[0] for call in run.call_args_list]
        packages = [command[-1] for command in commands if command[0] == "bash"]
        self.assertEqual(packages, ["libcamera", "libcamera_apps"])
        self.assertIn(["rpicam-still", "--version"], commands)
        self.assertIn(["rpicam-vid", "--version"], commands)

    def test_incomplete_package_installation_is_reported(self):
        with patch.object(setup, "run"), \
                patch.object(setup, "urlopen", return_value=io.BytesIO(b"#!/bin/bash\nexit 0\n")), \
                patch.object(setup.subprocess, "run", return_value=Mock(stdout="Unconfigured packages\n")), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(setup.SetupError, "incomplete"):
                setup.install_camera_stack()


if __name__ == "__main__":
    unittest.main()
