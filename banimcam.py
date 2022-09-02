from manim.camera.camera import Camera

class BanimCam(Camera):
    def capture_mobjects(self, mobjects, **kwargs):
        return super().capture_mobjects(mobjects, **kwargs)

    display_funcs = None

    def get_cairo_context(self, pixel_array):
        #Must check for a cached context,maybe leave that part the same
        #
        return super().get_cairo_context(pixel_array)

    def display_multiple_vectorized_mobjects(self, vmobjects, pixel_array):
        return super().display_multiple_vectorized_mobjects(vmobjects, pixel_array)

    def display_multiple_background_colored_vmobjects(self, cvmobjects, pixel_array):
        return super().display_multiple_background_colored_vmobjects(cvmobjects, pixel_array)

    def display_multiple_non_background_colored_vmobjects(self, vmobjects, pixel_array):
        return super().display_multiple_non_background_colored_vmobjects(vmobjects, pixel_array)

    def display_vectorized(self, vmobject, ctx):
        return super().display_vectorized(vmobject, ctx)

    def set_cairo_context_color(self, ctx, rgbas, vmobject):
        return super().set_cairo_context_color(ctx, rgbas, vmobject)

    def apply_fill(self, ctx, vmobject):
        return super().apply_fill(ctx, vmobject)

    def apply_stroke(self, ctx, vmobject, background=False):
        return super().apply_stroke(ctx, vmobject, background)

    def display_multiple_background_colored_vmobjects(self, cvmobjects, pixel_array):
        return super().display_multiple_background_colored_vmobjects(cvmobjects, pixel_array)

    