export interface INode {
  id: string;
  name: string;
  description: string;
}

export interface NodeGroup {
  name: string;
  nodes: INode[];
}

export interface NodeLibrary {
  groups: NodeGroup[];
}
