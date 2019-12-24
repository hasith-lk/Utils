from pywinauto.application import Application
# Run a target application
app = Application().start("notepad.exe")
# Select a menu item
app.UntitledNotepad.menu_select("Help->About Notepad")
# Click on a button
app.AboutNotepad.OK.click()
# Type a text string
app.UntitledNotepad.Edit.type_keys("pywinauto Works!", with_spaces = True)


# Local Test
ifsApp = Application(backend="uia").start("//cmbgse31/APP9_DEP_CMB/IFSHome/instance/APP9_DEP_CMB/client/runtime/Ifs.Fnd.Explorer.exe")

app = ifsApp.window(best_match ='IFS ApplicationsDialog')

ifsApp.Properties.print_control_identifiers()

#ifsApp.UntitledNotepad.menu_select("Help->About...")

app2 = Application().start('"C:\\Program Files (x86)\\Notepad++\\notepad++.exe"')
app2.