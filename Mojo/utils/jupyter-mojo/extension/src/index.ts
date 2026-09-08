//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

import {JupyterFrontEnd, JupyterFrontEndPlugin} from '@jupyterlab/application';
import {ICodeMirror} from '@jupyterlab/codemirror';
import {LabIcon} from '@jupyterlab/ui-components';

import MOJO_FIRE_SVG from '../style/logo.svg';

import {defineCodeMirrorMode} from './cmMode';

/**
 * The Mojo fire icon.
 */
export const mojoFireIcon = new LabIcon({
  name : 'mojo_jupyter:file-mojo-fire',
  svgstr : MOJO_FIRE_SVG,
});

/**
 * Initialization data for the mojo extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id : 'mojo_jupyter',
  autoStart : true,
  requires : [ ICodeMirror ],
  activate : (app: JupyterFrontEnd, codeMirror: ICodeMirror) => {
    // Define a codemirror mode for Mojo so that we can get syntax highlighting.
    defineCodeMirrorMode(codeMirror);

    // Define a file type for mojo, so that mojo files in Jupyter get handled
    // correctly.
    app.docRegistry.addFileType({
      name : 'mojo',
      mimeTypes : [ 'text/x-mojo' ],
      extensions : [ '.mojo' ],
      displayName : 'Mojo',
      icon : mojoFireIcon,
      contentType : 'file',
      fileFormat : 'text',
    });
  }
};

export default plugin;
