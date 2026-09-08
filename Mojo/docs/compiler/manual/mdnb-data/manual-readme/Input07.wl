snippet = SnippetSyntaxHighlight[
	{FileNameJoin[{$Modular, "KGEN"}], "KGENEnums.td"},
	"def KGEN_POCAttr : I32EnumAttr<\"POC\",",
    "PlainText",
    "ContextLines" -> 10
];
CellPrint@TextCell[
	snippet,
	"Program",
	TaggingRules -> <|
		"ConnorGray/Markdown" -> <|"CodeBlockInfoString" -> ""|>
	|>
]