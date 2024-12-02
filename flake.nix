{
  description = "Magic development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-compat = {
      url = "github:edolstra/flake-compat";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, flake-compat }:
    let
      linuxPkgs = nixpkgs.legacyPackages."x86_64-linux";
      darwinPkgs = nixpkgs.legacyPackages."aarch64-darwin";
      forAllSystems = function:
        nixpkgs.lib.genAttrs [
          "x86_64-linux"
          "aarch64-darwin"
        ] (system: function nixpkgs.legacyPackages.${system});
      version = "0.3.0";
      getBinary = system: {
          "x86_64-linux" = "magic-x86_64-unknown-linux-musl";
          "aarch64-darwin" = "magic-aarch64-apple-darwin";
      }.${system} or (throw "unsupported system: ${system}");
      fetchMagic = pkgs: binary: pkgs.stdenv.mkDerivation rec {
          name = "magic";
          src = pkgs.fetchurl {
            url = "https://dl.modular.com/public/magic/raw/versions/${version}/${binary}";
            sha256 = "sha256-wNweaK1TPqsVv9D1x7/m5kr1tryWfftzKCKDNdVBkOc=";
          };

          dontUnpack = true;

          postInstall = ''
            mkdir -p $out/bin
            cp $src $out/bin/magic
            chmod +x $out/bin/magic
          '';
        };
      magicEnv = pkgs: magic: (pkgs.buildFHSEnv {
        name = "magic-shell";
        targetPkgs = pkgs: (with pkgs; [
          libz
          clang
          lit
          llvm
          # Magic provides currently ncurses 6.5 on it's own but libtinfo is
          # needed and in nixpkgs this does not seem to be provided separately
          # https://github.com/NixOS/nixpkgs/issues/89769
          ncurses
        ]);
        profile = ''
          MODULAR_HOME=$HOME/.modular
          BIN_DIR=$MODULAR_HOME/bin
          MAGIC=$BIN_DIR/magic

          if [ ! -e $MAGIC ]; then
            mkdir -p $BIN_DIR
            cp ${magic}/bin/magic $BIN_DIR/magic
            echo 37e8aeee-7585-494b-83e4-59244188c2fe > "$MODULAR_HOME/webUserId"
          fi

          export PATH=$BIN_DIR:$PATH
          # Seems to work but prints error 'complete: command not found' when 
          # the shell is entered.
          # Maybe related to https://github.com/NixOS/nix/issues/6091#issuecomment-1038247010
          eval "$(magic completion --shell bash)"
        '';
        runScript = ''
          magic shell
        '';
      }).env;
    in
    {
      devShells = forAllSystems (pkgs: {
        default = magicEnv pkgs (fetchMagic pkgs (getBinary pkgs.system));
      });
    };
}

