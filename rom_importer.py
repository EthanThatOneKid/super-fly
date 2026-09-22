import os
import sys
import shutil


def _stable_retro():
    """Import stable-retro lazily so importing this module never requires the emulator.

    The experiment and evaluation plumbing (macro decoding, DAgger datasets,
    offline pipeline validation) must be importable on machines without the
    stable-retro native build; only the calls that actually touch the emulator
    pay the import cost.
    """
    import stable_retro
    return stable_retro


def import_nes_rom(rom_path: str, game_name: str = "SuperMarioBros-Nes") -> bool:
    """
    Imports custom NES ROM file into stable-retro library environment.
    """
    if not os.path.exists(rom_path):
        raise FileNotFoundError(f"Specified ROM file not found at path: {rom_path}")

    print(f"Importing ROM file '{rom_path}' for stable-retro game '{game_name}'...")

    stable_retro = _stable_retro()

    # Copy ROM file into retro's data path for game_name
    try:
        data_dir = stable_retro.data.path()
        game_dir = os.path.join(data_dir, "stable", game_name)
        if not os.path.exists(game_dir):
            game_dir = os.path.join(data_dir, "contrib", game_name)

        if os.path.exists(game_dir):
            target_rom = os.path.join(game_dir, "rom.nes")
            shutil.copyfile(rom_path, target_rom)
            print(f"Successfully copied ROM to: {target_rom}")
            return True
        else:
            # Attempt general retro ROM import via retro CLI entrypoint
            stable_retro.data.merge(rom_path)
            print("Successfully imported ROM via stable_retro.data.merge()")
            return True
    except Exception as e:
        print(f"Direct import note: {e}")
        # Try copying into current directory as fallback
        return False
