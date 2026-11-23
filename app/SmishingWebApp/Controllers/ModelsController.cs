using Microsoft.AspNetCore.Mvc;

namespace SmishingWebApp.Controllers;

public class ModelsController : Controller
{
    [HttpGet]
    public IActionResult Index()
    {
        return View();
    }
}