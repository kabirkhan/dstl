import catalogue


class registry:
    translators = catalogue.create("dstl", "translators", entry_points=True)


# class translator:
#     def __init__(self, name: str):
#         """Decorator for a translator

#         Args:
#             name (str): Operation name.
#             pre (Union[List[str], List[PreProcessor]]): List of preprocessors to run
#         """
#         self.name = name
#         self.pre = pre
#         self.handles_tokens = handles_tokens
#         self.factory = factory
#         self.augmentation = augmentation

#     def __call__(self, *args: Any, **kwargs: Any) -> Callable:
#         """Decorator for an operation.
#         The first arg is the function being decorated.
#         This function can either operate on a List[Example]
#         and in that case self.batch should be True.

#         e.g. @operation("recon.v1.some_name", batch=True)

#         Or it should operate on a single example and
#         recon will take care of applying it to a full Dataset

#         Args:
#             args: First arg is function to decorate

#         Returns:
#             Callable: Original function
#         """
#         op: Callable = args[0]

#         pre: List[PreProcessor] = []

#         for pre_name_or_op in self.pre:
#             preprocessor = pre_name_or_op
#             if isinstance(preprocessor, str):
#                 preprocessor = pre_registry.preprocessors.get(pre_name_or_op)
#             assert isinstance(preprocessor, PreProcessor)
#             pre.append(preprocessor)
