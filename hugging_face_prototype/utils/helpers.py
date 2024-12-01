def verify_input(question, options):
  while True:
    ans = input(question)
    if ans in options:
      return ans
    
def output_seperators(label, content):
  print("-" * 10)
  input("\n" + label)
  print(content)
  print("\n" + ("-" * 10))
